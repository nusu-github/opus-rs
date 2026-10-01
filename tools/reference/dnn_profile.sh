# Shared model arithmetic profile for the independent C neural oracles.
# The standard C build defines DISABLE_DEBUG_FLOAT; the diagnostic profile
# retains full-precision copies of layers that also have int8 weights.
case ${OPUS_DNN_DEBUG_FLOAT:-0} in
    0) dnn_flags=(-DDISABLE_DEBUG_FLOAT); dnn_suffix=-quantized ;;
    1) dnn_flags=(); dnn_suffix= ;;
    *) printf '%s\n' 'OPUS_DNN_DEBUG_FLOAT must be 0 or 1.' >&2; exit 2 ;;
esac
