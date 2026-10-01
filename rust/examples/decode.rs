use opus_rs::oggreader::{OggRead, OggReader, OggReaderError, ReadError};
use opus_rs::{Channels, Decoder, OpusDecodeError, OpusDecoderInitError};
use std::env;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::process;

const OPUS_TAGS_SIGNATURE: &[u8] = b"OpusTags";
const MAX_FRAME_SAMPLES: usize = 5760;

struct FileStream {
    file: File,
}

impl FileStream {
    fn new(file: File) -> Self {
        Self { file }
    }
}

impl OggRead for FileStream {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, ReadError> {
        use std::io::Read;

        loop {
            match self.file.read(buf) {
                Ok(n) => return Ok(n),
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(0),
                Err(_) => return Err(ReadError::Other),
            }
        }
    }
}

fn main() {
    if let Err(err) = run() {
        report_error(err);
        process::exit(1);
    }
}

fn run() -> Result<(), ExampleError> {
    let mut args = env::args_os();
    let _program = args.next();
    let input = args.next().ok_or(ExampleError::Usage)?;
    let output = args.next().ok_or(ExampleError::Usage)?;

    if args.next().is_some() {
        return Err(ExampleError::Usage);
    }

    let input_path = Path::new(&input);
    let output_path = Path::new(&output);

    let input_file =
        File::open(input_path).map_err(|err| ExampleError::Io("open input", err.kind()))?;
    let (mut ogg_reader, header) =
        OggReader::new_with(FileStream::new(input_file)).map_err(ExampleError::Ogg)?;

    let mut output_file =
        File::create(output_path).map_err(|err| ExampleError::Io("create output", err.kind()))?;
    let channels = match header.channels {
        1 => Channels::Mono,
        2 => Channels::Stereo,
        count => return Err(ExampleError::Channels(count)),
    };
    let mut decoder = Decoder::new(48_000, channels).map_err(ExampleError::Init)?;
    let mut pcm = vec![0i16; MAX_FRAME_SAMPLES * channels.count()];
    let mut skip = usize::from(header.pre_skip);

    loop {
        let (segments, _) = match ogg_reader.parse_next_page() {
            Ok(result) => result,
            Err(OggReaderError::Read(ReadError::UnexpectedEof)) => break,
            Err(err) => return Err(ExampleError::Ogg(err)),
        };

        if let Some(first) = segments.get(0)
            && first.starts_with(OPUS_TAGS_SIGNATURE)
        {
            continue;
        }

        for segment in segments.into_iter() {
            if segment.is_empty() {
                continue;
            }

            let samples = decoder
                .decode(segment, &mut pcm, false)
                .map_err(ExampleError::Decoder)?;
            let skipped = skip.min(samples);
            skip -= skipped;
            let bytes: Vec<u8> = pcm[skipped * channels.count()..samples * channels.count()]
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect();
            output_file
                .write_all(&bytes)
                .map_err(|err| ExampleError::Io("write output", err.kind()))?;
        }
    }

    Ok(())
}

fn report_error(err: ExampleError) {
    match err {
        ExampleError::Usage => {
            eprintln!("Usage: decode <in-file> <out-file>");
        }
        ExampleError::Io(context, kind) => {
            eprintln!("IO error ({context}): {kind:?}");
        }
        ExampleError::Ogg(err) => {
            eprintln!("ogg reader error: {err}");
        }
        ExampleError::Decoder(err) => {
            eprintln!("decoder error: {err:?}");
        }
        ExampleError::Init(err) => {
            eprintln!("decoder initialization error: {err:?}");
        }
        ExampleError::Channels(count) => {
            eprintln!("this example supports mono and stereo, received {count} channels");
        }
    }
}

enum ExampleError {
    Usage,
    Io(&'static str, io::ErrorKind),
    Ogg(OggReaderError),
    Decoder(OpusDecodeError),
    Init(OpusDecoderInitError),
    Channels(u8),
}

impl From<OggReaderError> for ExampleError {
    fn from(value: OggReaderError) -> Self {
        Self::Ogg(value)
    }
}

impl From<OpusDecodeError> for ExampleError {
    fn from(value: OpusDecodeError) -> Self {
        Self::Decoder(value)
    }
}
