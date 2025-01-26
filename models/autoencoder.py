import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_shape=(3, 64, 128)):  # size of images ارتفاع 64، عرض 128
        super(Autoencoder, self).__init__()
        self.input_shape = input_shape

        # --------------------------
        #        encoder انکودر
        # --------------------------
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # خروجی: (64, 32, 64)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # خروجی: (128, 16, 32)
        )

        #automaticly calculate size of images- محاسبه خودکار اندازه تخت‌شده
        self.flattened_size = self._calculate_flattened_size()
        self.encoder_fc = nn.Linear(self.flattened_size, 256)

        # --------------------------
        #        decoder دیکودر
        # --------------------------
        self.decoder = nn.Sequential(
            nn.Linear(256, self.flattened_size),
            nn.Unflatten(1, (128, 16, 32)),  #same as encoder output- مطابق خروجی انکودر

            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2,
                padding=1, output_padding=(1, 1)
            ),
            nn.ReLU(),  #  size of outputخروجی: (64, 32, 64)

            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2,
                padding=1, output_padding=(1, 1)
            ),
            nn.ReLU(),  #size of output- خروجی: (32, 64, 128)

            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # output- خروجی: (3, 64, 128)
        )

    def _calculate_flattened_size(self):
        dummy_input = torch.randn(1, *self.input_shape)
        with torch.no_grad():
            output = self.encoder_conv(dummy_input)
            return output.numel() // output.shape[0]

    def forward(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_fc(x)
        x = self.decoder(x)
        return x


# تست مدل
if __name__ == "__main__":
    model = Autoencoder(input_shape=(3, 64, 128))  #set the upper side ورودی نیمه بالایی
    dummy_input = torch.randn(16, 3, 64, 128)  # size of output-ابعاد: (64, 128)
    output = model(dummy_input)
    print(f"ورودی: {dummy_input.shape}")
    print(f"خروجی: {output.shape}")  #output size- باید (16, 3, 64, 128) باشد






