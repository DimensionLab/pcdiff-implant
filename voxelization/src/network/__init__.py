from src.network import decoder, encoder

encoder_dict = {
    "local_pool_pointnet": encoder.LocalPoolPointnet,
}
decoder_dict = {
    "simple_local": decoder.LocalDecoder,
}
