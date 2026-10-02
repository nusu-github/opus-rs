use opus_rs::oggreader::{OggRead, OggReader, OggReaderError, ReadError};
use opus_rs::{Channels, Decoder};

pub const MAX_PCM_SAMPLES: usize = 5760 * 2;
pub const OPUS_TAGS_SIGNATURE: &[u8] = b"OpusTags";
pub const TINY_OGG: &[u8] = include_bytes!("../fixtures/reference/celt-stereo-20ms.ogg");

pub struct SliceReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> SliceReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }
}

impl<'a> OggRead for SliceReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, ReadError> {
        if self.position >= self.data.len() {
            return Ok(0);
        }

        let remaining = self.data.len() - self.position;
        let to_copy = remaining.min(buf.len());
        buf[..to_copy].copy_from_slice(&self.data[self.position..self.position + to_copy]);
        self.position += to_copy;
        Ok(to_copy)
    }
}

pub fn fuzz_decoder(data: &[u8]) {
    let _ = decode_stream(data);
}

pub fn decode_stream(data: &[u8]) -> Result<usize, ()> {
    let mut out = [0i16; MAX_PCM_SAMPLES];

    let (mut ogg, header) = OggReader::new_with(SliceReader::new(data)).map_err(|_| ())?;
    let channels = match header.channels {
        1 => Channels::Mono,
        2 => Channels::Stereo,
        _ => return Err(()),
    };
    let mut decoder = Decoder::new(48_000, channels).map_err(|_| ())?;
    let mut decoded_frames = 0usize;

    loop {
        let (segments, _) = match ogg.parse_next_page() {
            Ok(result) => result,
            Err(OggReaderError::Read(ReadError::UnexpectedEof)) => break,
            Err(_) => return Err(()),
        };

        if let Some(first_segment) = segments.get(0)
            && first_segment.starts_with(OPUS_TAGS_SIGNATURE)
        {
            continue;
        }

        for segment in segments.into_iter() {
            if segment.is_empty() {
                continue;
            }

            match decoder.decode(segment, &mut out, false) {
                Ok(_) => decoded_frames += 1,
                Err(_) => return Err(()),
            }
        }
    }

    Ok(decoded_frames)
}
