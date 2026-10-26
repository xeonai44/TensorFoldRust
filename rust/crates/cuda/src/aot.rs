//! A captured set of Triton cubins (aot.json + cubins/): each launch
//! picks the variant Triton itself would have picked.
//!
//! Port of `zig/src/cuda/aot.zig` (210 lines): the variant-matching
//! whole — the JSON shape Triton's exporter writes, the specialization
//! rules (a pointer's 16-divisibility, an int folded to the constexpr 1,
//! the types named as Triton names them), and the launch in the
//! variant's own parameter order.

use crate::abi;
use crate::driver::Driver;
use crate::launch::{Args, Dim3};
use crate::stream::Stream;
use crate::triton::{Kernel, Meta};
use serde::Deserialize;
use std::collections::HashMap;

/// Where a family's glue kernels come from: our own fatbins, or a
/// captured set in a folder. Upstream: `Pick` (aot.zig:9).
pub enum Pick {
    Native,
    Captured(String),
    Missing(String),
}

/// `explicit` (folder or native), else the binary's
/// share/tensorfold/cuda/sm<capability> when it has aot.json. Upstream:
/// `pick` (aot.zig:15).
pub fn pick(explicit: Option<&str>, capability: u32) -> Pick {
    if let Some(dir) = explicit {
        if dir == "native" {
            return Pick::Native;
        }
        return if has(dir) {
            Pick::Captured(dir.into())
        } else {
            Pick::Missing(dir.into())
        };
    }
    let Ok(exe) = std::env::current_exe() else {
        return Pick::Native;
    };
    let Some(dir) = exe.parent() else {
        return Pick::Native;
    };
    let share = dir
        .join("../share/tensorfold/cuda")
        .join(format!("sm{capability}"));
    if has(share.to_str().unwrap_or_default()) {
        Pick::Captured(share.to_string_lossy().into_owned())
    } else {
        Pick::Native
    }
}

fn has(dir: &str) -> bool {
    std::path::Path::new(dir).join("aot.json").is_file()
}

/// The aot.json shapes. Upstream: the `*Json` structs (aot.zig:30-43).
#[derive(Deserialize)]
pub struct ParamJson {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(default)]
    pub div16: bool,
    #[serde(default)]
    pub nospec: bool,
}

#[derive(Deserialize, Default)]
pub struct ConstJson {
    #[serde(default)]
    pub int: Option<i64>,
    /// The f32's bits, as Triton's exporter writes them.
    #[serde(default)]
    pub f32: Option<u32>,
}

#[derive(Deserialize)]
pub struct KernelJson {
    #[serde(rename = "fn")]
    pub function: String,
    pub hash: String,
    pub name: String,
    pub num_warps: u32,
    #[serde(default = "default_one")]
    pub num_ctas: u32,
    #[serde(default)]
    pub shared: u32,
    #[serde(default)]
    pub global_scratch: u32,
    #[serde(default = "default_one")]
    pub global_align: u32,
    #[serde(default)]
    pub profile_scratch: u32,
    #[serde(default)]
    pub pdl: bool,
    #[serde(default)]
    pub params: Vec<ParamJson>,
    #[serde(default)]
    pub consts: HashMap<String, ConstJson>,
}

fn default_one() -> u32 {
    1
}

#[derive(Deserialize)]
struct SetJson {
    kernels: Vec<KernelJson>,
}

/// One argument of a launch, by the kernel's parameter name. Upstream:
/// `Arg` (aot.zig:50).
#[derive(Clone, Copy)]
pub struct Arg {
    pub name: &'static str,
    pub value: ArgValue,
}

#[derive(Clone, Copy)]
pub enum ArgValue {
    /// A pointer with its Triton type name.
    Ptr { addr: u64, ty: &'static str },
    I32(i32),
    F32(f32),
    U64(u64),
}

/// A constexpr the call site compiled the kernel with (Python's keyword
/// arguments): ints, bools, fp32 bits. Upstream: `Const` (aot.zig:57).
#[derive(Clone, Copy, Default)]
pub struct Const {
    pub name: &'static str,
    pub int: Option<i64>,
    pub f32: Option<f32>,
}

/// Upstream: the constructors (aot.zig:61-76).
pub fn ptr(name: &'static str, ty: &'static str, addr: u64) -> Arg {
    Arg { name, value: ArgValue::Ptr { addr, ty } }
}
pub fn int(name: &'static str, v: i32) -> Arg {
    Arg { name, value: ArgValue::I32(v) }
}
pub fn float(name: &'static str, v: f32) -> Arg {
    Arg { name, value: ArgValue::F32(v) }
}
pub fn word(name: &'static str, v: u64) -> Arg {
    Arg { name, value: ArgValue::U64(v) }
}
pub fn ci(name: &'static str, v: i64) -> Const {
    Const { name, int: Some(v), f32: None }
}
pub fn cf(name: &'static str, v: f32) -> Const {
    Const { name, int: None, f32: Some(v) }
}

/// Upstream: ``Set`` (aot.zig:85) — the parsed aot.json with every
/// cubin loaded into its own module.
pub struct Set<'d> {
    pub specs: Vec<KernelJson>,
    pub kernels: Vec<Kernel<'d>>,
}

impl<'d> Set<'d> {
    /// Loads every cubin listed in `dir`/aot.json into its own module.
    /// Upstream: `load` (aot.zig:88).
    pub fn load(
        d: &'d Driver,
        device: abi::Device,
        dir: &str,
    ) -> Result<Set<'d>, crate::driver::Error> {
        use crate::driver::Error;
        let text = std::fs::read_to_string(std::path::Path::new(dir).join("aot.json"))
            .map_err(|_| Error::Invalid)?;
        let parsed: SetJson = serde_json::from_str(&text).map_err(|_| Error::Invalid)?;
        let mut specs = Vec::with_capacity(parsed.kernels.len());
        let mut kernels = Vec::with_capacity(parsed.kernels.len());
        for k in parsed.kernels {
            if k.global_scratch != 0 || k.profile_scratch != 0 {
                return Err(Error::Invalid); // ScratchUnsupported
            }
            let file = std::path::Path::new(dir)
                .join("cubins")
                .join(format!("{}.cubin", k.hash));
            let cubin = std::fs::read(&file).map_err(|_| Error::Invalid)?;
            let meta = Meta {
                name: k.name.clone(),
                num_warps: k.num_warps,
                num_ctas: k.num_ctas,
                shared: k.shared,
                launch_pdl: k.pdl,
                ..Default::default()
            };
            let kernel = Kernel::load(d, device, &cubin, meta, &k.name)?;
            specs.push(k);
            kernels.push(kernel);
        }
        Ok(Set { specs, kernels })
    }

    /// The variant of `function` compiled for these constexprs and these
    /// arguments' specialization. Upstream: `find` (aot.zig:118).
    pub fn find(
        &self,
        function: &str,
        args: &[Arg],
        consts: &[Const],
    ) -> Result<usize, crate::driver::Error> {
        use crate::driver::Error;
        for (i, spec) in self.specs.iter().enumerate() {
            if spec.function != function {
                continue;
            }
            if matches_spec(spec, args, consts) {
                return Ok(i);
            }
        }
        eprintln!("no captured Triton variant of {function} for this launch:");
        for a in args {
            match a.value {
                ArgValue::Ptr { addr, ty } => eprintln!(
                    "  {}: {} at {addr:x} (16-aligned {})",
                    a.name, ty, addr % 16 == 0
                ),
                ArgValue::I32(x) => eprintln!("  {}: i32 {x}", a.name),
                ArgValue::F32(x) => eprintln!("  {}: fp32 {x}", a.name),
                ArgValue::U64(x) => eprintln!("  {}: u64 {x}", a.name),
            }
        }
        Err(Error::Invalid) // MissingTritonVariant
    }

    /// The smallest value of constexpr `name` at or above `at_least`
    /// among `function`'s variants. Upstream: `smallestConst` (aot.zig:131).
    pub fn smallest_const(&self, function: &str, name: &str, at_least: i64) -> Option<i64> {
        let mut best: Option<i64> = None;
        for spec in &self.specs {
            if spec.function != function {
                continue;
            }
            let Some(c) = spec.consts.get(name) else { continue };
            let Some(v) = c.int else { continue };
            if v >= at_least && best.is_none_or(|b| v < b) {
                best = Some(v);
            }
        }
        best
    }

    /// Launches `function` on `grid` as Triton's launcher would: runtime
    /// arguments in the variant's order, scratch null. Upstream: `run`
    /// (aot.zig:141).
    pub fn run(
        &self,
        stream: &Stream,
        function: &str,
        grid: [u32; 3],
        args: &[Arg],
        consts: &[Const],
    ) -> Result<(), crate::driver::Error> {
        let v = self.find(function, args, consts)?;
        let mut packed = Args::new();
        for p in &self.specs[v].params {
            let a = lookup(args, &p.name).ok_or(crate::driver::Error::Invalid)?;
            match a.value {
                ArgValue::Ptr { addr, .. } => packed.add(addr),
                ArgValue::I32(x) => packed.add(x),
                ArgValue::F32(x) => packed.add(x),
                ArgValue::U64(x) => packed.add(x),
            }
        }
        self.kernels[v].launch_on(
            Dim3 { x: grid[0], y: grid[1], z: grid[2] },
            stream,
            &mut packed,
            crate::triton::Scratch::default(),
            &[],
        )
    }
}

fn lookup(args: &[Arg], name: &str) -> Option<Arg> {
    args.iter().copied().find(|a| a.name == name)
}

/// The specialization rules (aot.zig:164's `matches`): the constexprs
/// equal, every runtime argument typed and divisibility-matched the way
/// the capture specialized them, an int Triton folded only ever the
/// value 1, and the runtime count exactly the kernel's parameters.
fn matches_spec(k: &KernelJson, args: &[Arg], consts: &[Const]) -> bool {
    for c in consts {
        let Some(got) = k.consts.get(c.name) else { return false };
        if let Some(x) = c.int {
            if got.int != Some(x) {
                return false;
            }
        }
        if let Some(x) = c.f32 {
            if got.f32 != Some(x.to_bits()) {
                return false;
            }
        }
    }
    let mut runtime: usize = 0;
    for a in args {
        let param = k.params.iter().find(|p| p.name == a.name);
        match a.value {
            ArgValue::I32(x) => {
                let Some(p) = param else {
                    // an int Triton folded: only the value 1 is ever
                    // specialized to a constexpr
                    let Some(got) = k.consts.get(a.name) else { return false };
                    if x != 1 || got.int != Some(1) {
                        return false;
                    }
                    continue;
                };
                if p.ty != "i32" {
                    return false;
                }
                if !p.nospec && x == 1 {
                    return false;
                }
                if p.div16 != (!p.nospec && x % 16 == 0) {
                    return false;
                }
            }
            ArgValue::Ptr { addr, ty } => {
                let Some(p) = param else { return false };
                if p.ty != ty || p.div16 != (addr % 16 == 0) {
                    return false;
                }
            }
            ArgValue::F32(_) => {
                let Some(p) = param else { return false };
                if p.ty != "fp32" {
                    return false;
                }
            }
            ArgValue::U64(x) => {
                let Some(p) = param else { return false };
                if p.ty != "u64" || p.div16 != (x % 16 == 0) {
                    return false;
                }
            }
        }
        runtime += 1;
    }
    runtime == k.params.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A spec from the JSON, for the match tests.
    fn spec(json: &str) -> KernelJson {
        serde_json::from_str::<KernelJson>(json).unwrap()
    }

    /// The specialization rules (aot.zig:164): the divisibility recorded
    /// at capture must match the argument now, an int the capture folded
    /// matches only the value 1, and the type names are Triton's own.
    #[test]
    fn the_specialization_rules() {
        // Two pointer params — one captured 16-divisible, one not
        let k = spec(
            r#"{"fn": "f", "hash": "h", "name": "f", "num_warps": 4,
                 "params": [
                   {"name": "a", "type": "*bf16", "div16": true},
                   {"name": "b", "type": "*fp32", "div16": false}
                 ]}"#,
        );
        // 16-divisible pointer + non-divisible pointer: matches
        assert!(matches_spec(
            &k,
            &[ptr("a", "*bf16", 256), ptr("b", "*fp32", 130)],
            &[]
        ));
        // The captured-divisible pointer given a non-divisible address: no
        assert!(!matches_spec(
            &k,
            &[ptr("a", "*bf16", 130), ptr("b", "*fp32", 130)],
            &[]
        ));
        // The wrong type name: no
        assert!(!matches_spec(
            &k,
            &[ptr("a", "*fp32", 256), ptr("b", "*fp32", 130)],
            &[]
        ));
        // A missing runtime argument: the count must match
        assert!(!matches_spec(&k, &[ptr("a", "*bf16", 256)], &[]));
        // An int Triton folded to the constexpr 1: matches only the value 1
        let k = spec(
            r#"{"fn": "f", "hash": "h", "name": "f", "num_warps": 4,
                 "params": [], "consts": {"n": {"int": 1}}}"#,
        );
        assert!(matches_spec(&k, &[int("n", 1)], &[]));
        assert!(!matches_spec(&k, &[int("n", 2)], &[]));
        // An i32 param the capture did NOT fold: the value 1 refuses
        // (Triton specializes 1 into a constexpr)
        let k = spec(
            r#"{"fn": "f", "hash": "h", "name": "f", "num_warps": 4,
                 "params": [{"name": "n", "type": "i32"}]}"#,
        );
        assert!(matches_spec(&k, &[int("n", 2)], &[]));
        assert!(!matches_spec(&k, &[int("n", 1)], &[]));
        // The div16 rule: the capture's record matches the argument now
        let k = spec(
            r#"{"fn": "f", "hash": "h", "name": "f", "num_warps": 4,
                 "params": [{"name": "n", "type": "i32", "div16": true}]}"#,
        );
        assert!(matches_spec(&k, &[int("n", 32)], &[]));
        assert!(!matches_spec(&k, &[int("n", 30)], &[]));
        // nospec: 1 allowed, div16 still recorded
        let k = spec(
            r#"{"fn": "f", "hash": "h", "name": "f", "num_warps": 4,
                 "params": [{"name": "n", "type": "i32", "nospec": true}]}"#,
        );
        assert!(matches_spec(&k, &[int("n", 1)], &[]));
    }

    /// The constexpr matches: the ints equal, the floats by bits.
    #[test]
    fn the_const_rules() {
        let k = spec(
            r#"{"fn": "f", "hash": "h", "name": "f", "num_warps": 4,
                 "params": [], "consts": {"BLOCK": {"int": 64}, "SCALE": {"f32": 1077936128}}}"#,
        );
        assert!(matches_spec(&k, &[], &[ci("BLOCK", 64)]));
        assert!(!matches_spec(&k, &[], &[ci("BLOCK", 128)]));
        assert!(matches_spec(&k, &[], &[cf("SCALE", 3.0)]), "the f32 bits");
        assert!(!matches_spec(&k, &[], &[cf("SCALE", 2.5)]));
        // A const the capture never took: no
        assert!(!matches_spec(&k, &[], &[ci("OTHER", 1)]));
    }

    /// `smallestConst`: the smallest int at or above the floor.
    #[test]
    fn smallest_const_finds_the_floor() {
        let set = Set {
            specs: vec![
                spec(r#"{"fn": "f", "hash": "a", "name": "f", "num_warps": 1, "consts": {"BLOCK": {"int": 32}}}"#),
                spec(r#"{"fn": "f", "hash": "b", "name": "f", "num_warps": 1, "consts": {"BLOCK": {"int": 128}}}"#),
                spec(r#"{"fn": "g", "hash": "c", "name": "g", "num_warps": 1, "consts": {"BLOCK": {"int": 8}}}"#),
            ],
            kernels: Vec::new(),
        };
        assert_eq!(set.smallest_const("f", "BLOCK", 64), Some(128));
        assert_eq!(set.smallest_const("f", "BLOCK", 16), Some(32));
        assert_eq!(set.smallest_const("g", "BLOCK", 4), Some(8));
        assert_eq!(set.smallest_const("g", "BLOCK", 16), None);
        assert_eq!(set.smallest_const("h", "BLOCK", 0), None);
    }
}
