use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

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
    jump_press_time_s: f32,
    mouse_sender: WindowMessageMouseSender,
    desired_pressed: bool,
    actual_pressed: bool,
    pending_use_release_at: Option<Instant>,
    pending_jump_release_at: Option<Instant>,
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
            jump_press_time_s: cfg.control.jump_press_time_s,
            mouse_sender: WindowMessageMouseSender::new(),
            desired_pressed: false,
            actual_pressed: false,
            pending_use_release_at: None,
            pending_jump_release_at: None,
        })
    }

    pub fn click_once(&mut self) -> Result<()> {
        self.pump_pending_actions()?;
        self.send_use(true)?;
        self.pending_use_release_at =
            Some(Instant::now() + Duration::from_millis(self.click_hold_ms));
        Ok(())
    }

    pub fn jump(&mut self) -> Result<()> {
        self.pump_pending_actions()?;
        let t = self.jump_press_time_s;
        if t <= 0.0 {
            return Ok(());
        }
        let d = Duration::from_secs_f32(t);

        self.send_button("/input/Jump", true)?;
        self.pending_jump_release_at = Some(Instant::now() + d);
        Ok(())
    }

    pub fn poll_focus(&mut self) -> Result<()> {
        self.pump_pending_actions()
    }

    pub fn set_chatbox_typing(&mut self, typing: bool) -> Result<()> {
        let packet = OscPacket::Message(OscMessage {
            addr: "/chatbox/typing".to_string(),
            args: vec![OscType::Bool(typing)],
        });
        self.rt
            .block_on(self.client.send_to_addr(packet, self.osc_target))?;
        Ok(())
    }

    pub fn send_chatbox_input(
        &mut self,
        text: &str,
        send_immediately: bool,
        trigger_sfx: bool,
    ) -> Result<()> {
        let packet = OscPacket::Message(OscMessage {
            addr: "/chatbox/input".to_string(),
            args: vec![
                OscType::String(text.to_string()),
                OscType::Bool(send_immediately),
                OscType::Bool(trigger_sfx),
            ],
        });
        self.rt
            .block_on(self.client.send_to_addr(packet, self.osc_target))?;
        Ok(())
    }

    pub fn set_press(&mut self, press: bool) -> Result<()> {
        self.desired_pressed = press;
        self.sync_mouse_state()
    }

    pub fn cancel_pending_actions(&mut self) -> Result<()> {
        if self.pending_use_release_at.take().is_some() {
            self.send_use(false)?;
        }
        if self.pending_jump_release_at.take().is_some() {
            self.send_button("/input/Jump", false)?;
        }
        Ok(())
    }

    pub fn pump_pending_actions(&mut self) -> Result<()> {
        let now = Instant::now();
        if self
            .pending_use_release_at
            .is_some_and(|release_at| now >= release_at)
        {
            self.send_use(false)?;
            self.pending_use_release_at = None;
        }
        if self
            .pending_jump_release_at
            .is_some_and(|release_at| now >= release_at)
        {
            self.send_button("/input/Jump", false)?;
            self.pending_jump_release_at = None;
        }
        Ok(())
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
        let _ = self.cancel_pending_actions();
        self.desired_pressed = false;
        let _ = self.sync_mouse_state();
        let _ = self.rt.block_on(self.client.shutdown());
    }
}

