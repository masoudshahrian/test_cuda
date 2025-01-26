import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_shape=(3, 128, 128)):
        super(Autoencoder, self).__init__()

        # --------------------------
        #         انکودر
        # --------------------------
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # اندازه خروجی: (64, H/2, W/2)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # اندازه خروجی: (128, H/4, W/4)
        )

        # محاسبه خودکار اندازه تخت‌شده (Flattened Size)
        self.flattened_size = self._calculate_flattened_size(input_shape)

        self.encoder_fc = nn.Linear(self.flattened_size, 256)

        # --------------------------
        #         دیکودر
        # --------------------------
        self.decoder = nn.Sequential(
            nn.Linear(256, self.flattened_size),
            nn.Unflatten(1, (128, input_shape[1] // 4, input_shape[2] // 4)),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _calculate_flattened_size(self, input_shape):
        """محاسبه خودکار اندازه خروجی انکودر"""
        dummy_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            output = self.encoder_conv(dummy_input)
            return output.numel() // output.shape[0]  # (تعداد ویژگی‌ها بر نمونه)

    def forward(self, x):
        # انکودر
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)  # تبدیل به بردار
        x = self.encoder_fc(x)

        # دیکودر
        x = self.decoder(x)
        return x


# --------------------------
# تست مدل با یک نمونه تصادفی
# --------------------------
if __name__ == "__main__":
    # پارامترهای ورودی (متناسب با داده‌های شما)
    BATCH_SIZE = 16
    CHANNELS = 3
    HEIGHT = 128  # تغییر به اندازه واقعی تصاویر شما
    WIDTH = 128  # تغییر به اندازه واقعی تصاویر شما

    # ساخت مدل
    model = Autoencoder(input_shape=(CHANNELS, HEIGHT, WIDTH))

    # ساخت یک ورودی نمونه
    dummy_input = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)

    # تست عملکرد مدل
    output = model(dummy_input)

    # چاپ ابعاد
    print(f"ورودی: {dummy_input.shape}")
    print(f"خروجی: {output.shape}")