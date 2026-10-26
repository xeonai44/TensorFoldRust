//! The objc plumbing the runtime Metal path needs: NSStrings.
//!
//! Kept to the minimum the port needs; the `metal` crate owns the typed wrappers
//! everywhere it has one. Upstream's own seam is `zig/src/metal/objc.zig`.

use ::objc::runtime::Object;
use ::objc::{class, msg_send, sel, sel_impl};

pub type Id = *mut Object;

/// An NSString made from UTF-8 text, the way upstream's `objc.string` makes one —
/// `alloc` + `initWithBytes:length:encoding:`, so the caller owns it and releases it
/// (upstream metal/objc.zig:82; a convenience constructor would hand back an
/// autoreleased string the caller must not release).
///
/// # Safety
///
/// The caller must release the returned object (or hand it to an API that takes
/// ownership).
pub unsafe fn string(text: &str) -> Id {
    unsafe {
        let raw: Id = msg_send![class!(NSString), alloc];
        // NSUTF8StringEncoding is 4.
        let string: Id = msg_send![
            raw,
            initWithBytes: text.as_ptr()
            length: text.len()
            encoding: 4usize
        ];
        if string.is_null() {
            let _: () = msg_send![raw, release];
            panic!("objc: string is not UTF-8");
        }
        string
    }
}

/// An NSString's text, for error messages.
pub(crate) unsafe fn nsstring_text(string: Id) -> String {
    unsafe {
        let utf8: *const std::os::raw::c_char = msg_send![string, UTF8String];
        std::ffi::CStr::from_ptr(utf8).to_string_lossy().into_owned()
    }
}

