//! The family's errors, one set as Zig's anyerror flows them.
//!
//! Every error name the nemotron modules raise, with the CUDA runtime's
//! carried through `Cuda` — upstream's error unions have no boundary
//! between the family's refusals and the driver's.

use tensorfold_cuda::Error as CudaError;

#[derive(Debug)]
pub enum Error {
    Cuda(CudaError),
    /// config.json is missing a field or holds a non-integer where an
    /// integer belongs.
    BadConfig,
    NotNemotronH,
    UnsupportedBlock,
    /// A quantization block the native kernels do not read (not MLX
    /// affine at the checkpoint's width).
    UnsupportedQuantization,
    MixedQuantization,
    /// The shapes differ from the kernels built for Nemotron 3.5
    /// Lightning 30B-A3B.
    UnsupportedShapes,
    DraftIdPastVocabulary,
    DraftIdsNotTiled,
    /// A norm row wider than the kernel's threads.
    RowTooWide,
    RouterTooWide,
    ScanShape,
    AttentionShape,
    /// Our keyed kernel takes the greedy rule only.
    KeyedRule,
    /// The decode plan kernel's fixed bounds exceeded.
    PlanTooWide,
    /// A tensor the loader did not expect (rank, dtype or shape).
    UnexpectedTensor,
    SharedExpertWidth,
    /// A tensor larger than a checkpoint-read slot.
    TensorLargerThanSlot,
    /// A page-locked slot that came back unaligned.
    UnalignedPinnedSlot,
    /// The checkpoint's read-side failures, carried from direct_io.
    FileNotFound,
    /// A carveout refusal, its own text.
    Carveout(String),
    UnusedCheckpointTensors,
    UnusedMtpTensors,
    NoMtpHead,
    BadSegments,
    ContextUnsupported,
    EmptyPrompt,
    BadReuse,
    ContextFull,
    BadWindow,
    WindowTooWide,
    BadKeep,
    Cancelled,
    BadDraftRequest,
    UnknownStream,
    PromptTooLong,
    NoSuchToken,
    EmptyPath,
    NoFirstToken,
    NoFirstRow,
    TreesNotBuilt,
    NotPipelined,
    NotAtMark,
    Backend(String),
    HeatBands,
    /// A checkpoint name that does not exist, its text.
    MissingTensor(String),
    BufferTooSmall,
    ReadFailed,
    EndOfFile,
}

impl From<tensorfold_core::checkpoint::MissingTensor> for Error {
    fn from(e: tensorfold_core::checkpoint::MissingTensor) -> Error {
        Error::MissingTensor(e.0)
    }
}

impl From<tensorfold_core::checkpoint::UnexpectedTensor> for Error {
    fn from(e: tensorfold_core::checkpoint::UnexpectedTensor) -> Error {
        Error::MissingTensor(e.0)
    }
}

/// The Linux carveout's own refusals; off Linux the host-only
/// Carveout refuses with the driver's error (carried by `Cuda`).
#[cfg(target_os = "linux")]
impl From<tensorfold_cuda::CarveoutError> for Error {
    fn from(e: tensorfold_cuda::CarveoutError) -> Error {
        // Upstream: the carveout's refusals flow as anyerror through
        // the state layer; the port keeps their text.
        Error::Carveout(e.to_string())
    }
}

impl From<tensorfold_core::heat::HeatError> for Error {
    fn from(e: tensorfold_core::heat::HeatError) -> Error {
        use tensorfold_core::heat::HeatError;
        match e {
            HeatError::HeatBands => Error::HeatBands,
            HeatError::Cancelled => Error::Cancelled,
            _ => Error::Cuda(tensorfold_cuda::Error::Invalid),
        }
    }
}

impl From<crate::uring::EnterError> for Error {
    fn from(_: crate::uring::EnterError) -> Error {
        Error::ReadFailed
    }
}

impl From<tensorfold_core::direct_io::ReadError> for Error {
    fn from(e: tensorfold_core::direct_io::ReadError) -> Error {
        use tensorfold_core::direct_io::ReadError;
        match e {
            ReadError::FileNotFound => Error::FileNotFound,
            ReadError::BufferTooSmall => Error::BufferTooSmall,
            ReadError::ReadFailed => Error::ReadFailed,
            ReadError::EndOfFile => Error::EndOfFile,
        }
    }
}

impl From<CudaError> for Error {
    fn from(e: CudaError) -> Error {
        Error::Cuda(e)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Cuda(e) => write!(f, "{e}"),
            Error::BadConfig => write!(f, "config.json does not hold this field"),
            Error::NotNemotronH => write!(f, "not a Nemotron-H checkpoint"),
            Error::UnsupportedBlock => write!(f, "a block kind the loader does not read"),
            Error::UnsupportedQuantization => {
                write!(f, "the native Nemotron kernels read MLX affine checkpoints")
            }
            Error::MixedQuantization => write!(f, "the checkpoint quantizes at mixed widths"),
            Error::UnsupportedShapes => write!(
                f,
                "this Nemotron-H's shapes differ from the kernels built for Nemotron 3.5 Lightning 30B-A3B"
            ),
            Error::DraftIdPastVocabulary => write!(f, "a draft id is past the vocabulary"),
            Error::DraftIdsNotTiled => write!(f, "the draft ids are not a multiple of 64"),
            Error::RowTooWide => write!(f, "the norm row is too wide for the kernel"),
            Error::RouterTooWide => write!(f, "the router is too wide for the kernel"),
            Error::ScanShape => write!(f, "the scan's shapes are not what the kernel serves"),
            Error::AttentionShape => write!(f, "the attention shapes are not what the kernel serves"),
            Error::KeyedRule => write!(f, "our keyed kernel takes the greedy rule"),
            Error::PlanTooWide => write!(f, "the plan kernel's bounds are exceeded"),
            Error::UnexpectedTensor => write!(f, "a tensor the loader did not expect"),
            Error::SharedExpertWidth => write!(f, "the shared expert's width is unexpected"),
            Error::TensorLargerThanSlot => write!(f, "the tensor is larger than a read slot"),
            Error::UnalignedPinnedSlot => write!(f, "the page-locked slot is unaligned"),
            Error::FileNotFound => write!(f, "the file does not open"),
            Error::Carveout(text) => write!(f, "{text}"),
            Error::UnusedCheckpointTensors => {
                write!(f, "the checkpoint holds tensors the loader never read")
            }
            Error::UnusedMtpTensors => {
                write!(f, "the MTP head holds tensors the loader never read")
            }
            Error::MissingTensor(name) => write!(f, "checkpoint has no tensor {name}"),
            Error::NoMtpHead => write!(f, "the checkpoint carries no MTP head"),
            Error::BadSegments => write!(f, "the segment count is out of range"),
            Error::ContextUnsupported => write!(f, "no captured kernel covers this context"),
            Error::EmptyPrompt => write!(f, "the prompt is empty"),
            Error::BadReuse => write!(f, "the state does not resume at this point"),
            Error::ContextFull => write!(f, "the prompt does not fit the context"),
            Error::BadWindow => write!(f, "the window is out of range"),
            Error::WindowTooWide => write!(f, "the window's rows are too wide"),
            Error::BadKeep => write!(f, "the keep is out of range"),
            Error::Cancelled => write!(f, "the request was cancelled"),
            Error::BadDraftRequest => write!(f, "the draft request is out of range"),
            Error::UnknownStream => write!(f, "the backend does not hold this stream"),
            Error::PromptTooLong => write!(f, "the prompt does not leave room for the reply"),
            Error::NoSuchToken => write!(f, "no drawn token at that handle"),
            Error::EmptyPath => write!(f, "a window kept nothing"),
            Error::NoFirstToken => write!(f, "the draft request has no first token"),
            Error::NoFirstRow => write!(f, "the stream has no first logprob row"),
            Error::TreesNotBuilt => write!(f, "the backend does not build trees"),
            Error::NotPipelined => write!(f, "the backend does not pipeline"),
            Error::NotAtMark => write!(f, "the engine does not stand at that mark"),
            Error::Backend(text) => write!(f, "{text}"),
            Error::HeatBands => write!(f, "the heat bands are not a pair"),
            Error::BufferTooSmall => write!(f, "the buffer is too small for the read"),
            Error::ReadFailed => write!(f, "the read failed"),
            Error::EndOfFile => write!(f, "the read ran past the end of the file"),
        }
    }
}

impl std::error::Error for Error {}
