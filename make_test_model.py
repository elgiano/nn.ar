import torch
import torch.nn as nn
import nn_tilde

class TinyCodec(nn_tilde.Module):
    def __init__(self):
        super(TinyCodec, self).__init__()
        bs = 16
        # Tiny deterministic encoder/decoder
        self.encoder = nn.Conv1d(1, 8, kernel_size=bs, stride=bs)# [B, 8, N//bs]
        self.decoder = nn.ConvTranspose1d(8, 1, kernel_size=bs, stride=bs)  # [B, 1, N]
        self.sr = 44100
        self.n_channels = self.target_channels = 1
        self.stereo_mode = False
        self.latent_size = 8

        x_len = bs*2
        x = torch.zeros(1, self.n_channels, x_len)
        z = self.encode(x)
        ratio_encode = x_len // z.shape[-1]

        self.register_method(
            "encode",
            in_channels=self.n_channels,
            in_ratio=1,
            out_channels=self.latent_size,
            out_ratio=ratio_encode,
            input_labels=['(signal) Channel %d'%d for d in range(1, self.n_channels+1)],
            output_labels=[
                f'(signal) Latent dimension {i + 1}'
                for i in range(self.latent_size)
            ],
        )
        self.register_method(
            "decode",
            in_channels=self.latent_size,
            in_ratio=ratio_encode,
            out_channels=self.target_channels,
            out_ratio=1,
            input_labels=[
                f'(signal) Latent dimension {i+1}'
                for i in range(self.latent_size)
            ],
            output_labels=['(signal) Channel %d'%d for d in range(1, self.target_channels+1)],
        )

        self.register_method(
            "forward",
            in_channels=self.n_channels,
            in_ratio=1,
            out_channels=self.target_channels,
            out_ratio=1,
            input_labels=['(signal) Channel %d'%d for d in range(1, self.n_channels + 1)],
            output_labels=['(signal) Channel %d'%d for d in range(1, self.target_channels+1)],
            test_buffer_size=64
        )

        self.test_attr = (False,)
        self.register_attribute("test_attr", False)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))

    @torch.jit.export
    def get_test_attr(self) -> bool:
        return self.test_attr[0]

    @torch.jit.export
    def set_test_attr(self, learn_target: bool) -> int:
        self.test_attr = (learn_target, )
        return 0


def save_model(path="nnar_test_model.ts"):
    model = TinyCodec()
    model.eval()
    model.export_to_ts(path)
    print(f"Model saved to {path}")


def test_model(path="nnar_test_model.ts"):
    m = torch.jit.load(path)
    example_input = torch.randn(1, 16)
    print("Original input:", example_input)
    encoded = m.encode(example_input)
    print("Encoded output:", encoded)
    decoded = m.decode(encoded)
    print("Decoded output:", decoded)
    forward_output = m(example_input)
    print("Forward output:", forward_output)

save_model()
test_model()

