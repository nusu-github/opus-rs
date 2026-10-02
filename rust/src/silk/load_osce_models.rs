//! Load optional speech enhancement models into the SILK decoder.
use super::dec_api::Decoder;
use super::errors::SilkError;

/// Load an owned copy of an Opus neural model blob.
/// Non-OSCE builds retain the reference no-op behavior.
pub fn load_osce_models(decoder: &mut Decoder, data: Option<&[u8]>) -> Result<(), SilkError> {
    #[cfg(feature = "osce")]
    {
        let result = data.ok_or(SilkError::DecPayloadError).and_then(|data| {
            crate::osce::OsceModel::from_bytes(data).map_err(|_| SilkError::DecPayloadError)
        });
        match result {
            Ok(model) => {
                decoder.osce_model = Some(alloc::sync::Arc::new(model));
                Ok(())
            }
            Err(error) => {
                decoder.osce_model = None;
                Err(error)
            }
        }
    }
    #[cfg(not(feature = "osce"))]
    {
        let _ = (decoder, data);
        Ok(())
    }
}

#[cfg(all(test, not(feature = "osce")))]
mod tests {
    use super::*;
    #[test]
    fn disabled_models_accept_any_payload() {
        let mut decoder = Decoder::default();
        assert_eq!(load_osce_models(&mut decoder, None), Ok(()));
        assert_eq!(load_osce_models(&mut decoder, Some(&[1, 2, 3, 4])), Ok(()));
    }
}
