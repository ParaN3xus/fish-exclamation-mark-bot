use std::mem::size_of;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use anyhow::{Result, bail};
use tokio::runtime::Runtime;
use vrchat_osc::VRChatOSC;
use vrchat_osc::rosc::{OscMessage, OscPacket, OscType};
use windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use windows::Win32::System::Threading::GetCurrentThreadId;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    INPUT, INPUT_0, INPUT_MOUSE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, MOUSEINPUT, SendInput,
};
use windows::Win32::UI::Accessibility::{
    HWINEVENTHOOK, SetWinEventHook, UnhookWinEvent,
};
use windows::Win32::UI::WindowsAndMessaging::{
    DispatchMessageW, EVENT_SYSTEM_FOREGROUND, GetForegroundWindow, GetMessageW, MSG,
    PostThreadMessageW, TranslateMessage,
    WINEVENT_OUTOFCONTEXT, WM_QUIT,
};
use tracing::warn;

use crate::config::AppConfig;
use crate::vrc_window::target_hwnd;

struct SendInputMouseSender;

impl SendInputMouseSender {
    fn new() -> Self {
        Self
    }

    fn set_lbutton(&self, press: bool) -> Result<()> {
        let flags = if press {
            MOUSEEVENTF_LEFTDOWN
        } else {
            MOUSEEVENTF_LEFTUP
        };
        let input = INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx: 0,
                    dy: 0,
                    mouseData: 0,
                    dwFlags: flags,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        };

        let sent = unsafe { SendInput(&[input], size_of::<INPUT>() as i32) };
        if sent != 1 {
            bail!("SendInput failed: sent={sent}");
        }
        Ok(())
    }
}

#[derive(Default)]
struct FocusShared {
    focused: AtomicBool,
    change_seq: AtomicU64,
}

impl FocusShared {
    fn set_focused(&self, focused: bool) {
        let old = self.focused.swap(focused, Ordering::Relaxed);
        if old != focused {
            self.change_seq.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn is_focused(&self) -> bool {
        self.focused.load(Ordering::Relaxed)
    }
}

static FOCUS_SHARED: OnceLock<Arc<FocusShared>> = OnceLock::new();

fn hwnd_is_target(hwnd: HWND) -> bool {
    let target = target_hwnd();
    !target.is_null() && hwnd.0 == target
}

fn foreground_is_target() -> bool {
    let hwnd = unsafe { GetForegroundWindow() };
    hwnd_is_target(hwnd)
}

unsafe extern "system" fn focus_event_proc(
    _h_win_event_hook: HWINEVENTHOOK,
    event: u32,
    hwnd: HWND,
    _id_object: i32,
    _id_child: i32,
    _dw_event_thread: u32,
    _dwms_event_time: u32,
) {
    if event != EVENT_SYSTEM_FOREGROUND {
        return;
    }
    if let Some(shared) = FOCUS_SHARED.get() {
        shared.set_focused(hwnd_is_target(hwnd));
    }
}

struct FocusWatcher {
    shared: Arc<FocusShared>,
    stop: Arc<AtomicBool>,
    thread_id: Arc<AtomicU32>,
    handle: Option<thread::JoinHandle<()>>,
}

impl FocusWatcher {
    fn new() -> Self {
        let shared = Arc::new(FocusShared::default());
        shared.set_focused(foreground_is_target());
        let _ = FOCUS_SHARED.set(shared.clone());

        let stop = Arc::new(AtomicBool::new(false));
        let thread_id = Arc::new(AtomicU32::new(0));

        let stop_c = stop.clone();
        let tid_c = thread_id.clone();
        let handle = thread::spawn(move || {
            let tid = unsafe { GetCurrentThreadId() };
            tid_c.store(tid, Ordering::Relaxed);

            let hook = unsafe {
                SetWinEventHook(
                    EVENT_SYSTEM_FOREGROUND,
                    EVENT_SYSTEM_FOREGROUND,
                    None,
                    Some(focus_event_proc),
                    0,
                    0,
                    WINEVENT_OUTOFCONTEXT,
                )
            };

            if hook.0.is_null() {
                warn!("SetWinEventHook failed");
            }

            while !stop_c.load(Ordering::Relaxed) {
                let mut msg = MSG::default();
                let ret = unsafe { GetMessageW(&mut msg, None, 0, 0) };
                if ret.0 <= 0 {
                    break;
                }
                unsafe {
                    let _ = TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            }

            if !hook.0.is_null() {
                let _ = unsafe { UnhookWinEvent(hook) };
            }
        });

        Self {
            shared,
            stop,
            thread_id,
            handle: Some(handle),
        }
    }

    fn is_focused(&self) -> bool {
        self.shared.is_focused()
    }

    fn refresh_now(&self) {
        self.shared.set_focused(foreground_is_target());
    }
}

impl Drop for FocusWatcher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let tid = self.thread_id.load(Ordering::Relaxed);
        if tid != 0 {
            let _ = unsafe { PostThreadMessageW(tid, WM_QUIT, WPARAM(0), LPARAM(0)) };
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

pub struct VrchatClicker {
    rt: Runtime,
    client: Arc<VRChatOSC>,
    osc_target: SocketAddr,
    click_hold_ms: u64,
    shake_head_time_s: f32,
    mouse_sender: SendInputMouseSender,
    focus: FocusWatcher,
    desired_pressed: bool,
    actual_pressed: bool,
}

impl VrchatClicker {
    pub fn new(cfg: &AppConfig) -> Result<Self> {
        let rt = Runtime::new()?;
        let client = rt.block_on(VRChatOSC::new(Some(IpAddr::V4(Ipv4Addr::LOCALHOST))))?;
        let osc_ip = cfg
            .control
            .osc_target_host
            .parse::<IpAddr>()
            .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST));
        let osc_target = SocketAddr::new(osc_ip, cfg.control.osc_target_port);
        Ok(Self {
            rt,
            client,
            osc_target,
            click_hold_ms: cfg.control.click_hold_ms,
            shake_head_time_s: cfg.control.shake_head_time_s,
            mouse_sender: SendInputMouseSender::new(),
            focus: FocusWatcher::new(),
            desired_pressed: false,
            actual_pressed: false,
        })
    }

    pub fn click_once(&mut self) -> Result<()> {
        self.send_use(true)?;
        thread::sleep(Duration::from_millis(self.click_hold_ms));
        self.send_use(false)?;
        Ok(())
    }

    pub fn shake_head(&self) -> Result<()> {
        let t = self.shake_head_time_s;
        if t <= 0.0 {
            return Ok(());
        }
        let d1 = Duration::from_secs_f32(t);
        let d2 = Duration::from_secs_f32((t * 2.0).max(0.0));

        self.send_button("/input/LookLeft", true)?;
        thread::sleep(d1);
        self.send_button("/input/LookLeft", false)?;

        self.send_button("/input/LookRight", true)?;
        thread::sleep(d2);
        self.send_button("/input/LookRight", false)?;

        self.send_button("/input/LookLeft", true)?;
        thread::sleep(d1);
        self.send_button("/input/LookLeft", false)?;
        Ok(())
    }

    pub fn poll_focus(&mut self) -> Result<()> {
        self.focus.refresh_now();
        self.sync_sendinput_state()
    }

    pub fn set_press(&mut self, press: bool) -> Result<()> {
        self.desired_pressed = press;
        self.sync_sendinput_state()
    }

    fn sync_sendinput_state(&mut self) -> Result<()> {
        if !self.focus.is_focused() {
            if self.actual_pressed {
                self.mouse_sender.set_lbutton(false)?;
                self.actual_pressed = false;
            }
            return Ok(());
        }

        if self.desired_pressed != self.actual_pressed {
            self.mouse_sender.set_lbutton(self.desired_pressed)?;
            self.actual_pressed = self.desired_pressed;
        }
        Ok(())
    }

    fn send_use(&self, press: bool) -> Result<()> {
        self.send_button("/input/UseRight", press)
    }

    fn send_button(&self, addr: &str, press: bool) -> Result<()> {
        let packet = OscPacket::Message(OscMessage {
            addr: addr.to_string(),
            args: vec![OscType::Int(if press { 1 } else { 0 })],
        });
        self.rt
            .block_on(self.client.send_to_addr(packet, self.osc_target))?;
        Ok(())
    }
}

impl Drop for VrchatClicker {
    fn drop(&mut self) {
        self.desired_pressed = false;
        let _ = self.sync_sendinput_state();
        let _ = self.rt.block_on(self.client.shutdown());
    }
}

