#[derive(Clone)]
pub struct FramePacket {
    pub w: i32,
    pub h: i32,
    pub bgra: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
pub struct BBox {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct Kp {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct OuterDet {
    pub b: BBox,
    pub top: Kp,
    pub bot: Kp,
    pub conf: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BotState {
    Stopped,
    WaitingFish,
    BiteOrError,
    Fishing,
    CollectFish,
    ReleaseLine,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectCommand {
    ForceFishComes,
    Reset,
    ToggleStateMachine,
}

#[derive(Clone)]
pub struct DetectPacket {
    pub t_sec: f64,
    pub w: i32,
    pub h: i32,
    pub bgr: Vec<u8>,
    pub state: BotState,
    pub press: bool,
    pub state_machine_enabled: bool,
    pub bite_similarity: f32,
    pub success_similarity: f32,
    pub fail_similarity: f32,
    pub collected_similarity: f32,
    pub bite_hit: bool,
    pub success_hit: bool,
    pub fail_hit: bool,
    pub collected_hit: bool,
    pub policy_fish_center: Option<f32>,
    pub policy_player_center: Option<f32>,
    pub policy_progress: Option<f32>,
    pub policy_target_half: Option<f32>,
    pub fps_cap: f32,
    pub fps_det: f32,
}

#[derive(Clone)]
pub struct TrackState {
    pub roi: BBox,
    pub outer_rel: BBox,
    pub top_rel: Kp,
    pub bot_rel: Kp,
    pub bright_h_norm: i32,
    pub fish_spec: Option<Vec<f32>>,
    pub proc_top: i32,
    pub proc_bot: i32,
}

