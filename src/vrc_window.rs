use std::sync::atomic::{AtomicIsize, Ordering};

static VRCHAT_HWND_BITS: AtomicIsize = AtomicIsize::new(0);

pub fn set_target_hwnd(hwnd: *mut core::ffi::c_void) {
    VRCHAT_HWND_BITS.store(hwnd as isize, Ordering::Relaxed);
}

pub fn clear_target_hwnd() {
    VRCHAT_HWND_BITS.store(0, Ordering::Relaxed);
}

pub fn target_hwnd() -> *mut core::ffi::c_void {
    VRCHAT_HWND_BITS.load(Ordering::Relaxed) as *mut core::ffi::c_void
}

