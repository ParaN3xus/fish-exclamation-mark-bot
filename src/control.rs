use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{Result, bail};
use tokio::runtime::Runtime;
use vrchat_osc::VRChatOSC;
use vrchat_osc::rosc::{OscMessage, OscPacket, OscType};
use windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::{
    PostMessageW, WM_LBUTTONDOWN, WM_LBUTTONUP,
};

use crate::config::AppConfig;
use crate::vrc_window::target_hwnd;

struct WindowMessageMouseSender;

impl WindowMessageMouseSender {
    fn new() -> Self {
        Self
    }

    fn set_lbutton(&self, press: bool) -> Result<()> {
        let hwnd_raw = target_hwnd();
        if hwnd_raw.is_null() {
            bail!("target hwnd is null");
        }
        let hwnd = HWND(hwnd_raw);

        let (msg, wparam) = if press {
            (WM_LBUTTONDOWN, WPARAM(1))
        } else {
            (WM_LBUTTONUP, WPARAM(0))
        };
        let lparam = LPARAM(0);

        if let Err(e) = unsafe { PostMessageW(Some(hwnd), msg, wparam, lparam) } {
            bail!("PostMessageW failed: msg={msg:#x}, err={e}");
        }
        Ok(())
    }
}

pub struct VrchatClicker {
    rt: Runtime,
    client: Arc<VRChatOSC>,
    osc_target: SocketAddr,
    click_hold_ms: u64,
    shake_head_time_s: f32,
    mouse_sender: WindowMessageMouseSender,
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
            mouse_sender: WindowMessageMouseSender::new(),
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
        Ok(())
    }

    pub fn set_press(&mut self, press: bool) -> Result<()> {
        self.desired_pressed = press;
        self.sync_mouse_state()
    }

    fn sync_mouse_state(&mut self) -> Result<()> {
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
        let _ = self.sync_mouse_state();
        let _ = self.rt.block_on(self.client.shutdown());
    }
}

