import os
import torch
import matplotlib.pyplot as plt

def save_model(model, path):
    """
    ذخیره مدل آموزش‌داده‌شده در فایل
    saving the trained model
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    """
    بارگذاری مدل ذخیره‌شده از فایل
    loading saved model
    """
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Model loaded from {path}")
    else:
        print(f"No model found at {path}")
    return model

def visualize_results(upper_half, lower_half, predictions, n=5):
    """
    نمایش تصاویر ورودی، خروجی واقعی و خروجی پیش‌بینی‌شده
    show the real image, real output and predicted output
    """
    for i in range(min(n, upper_half.size(0))):
        plt.subplot(3, n, i + 1)
        plt.imshow(upper_half[i].permute(1, 2, 0).numpy())
        plt.title("Input")
        plt.axis('off')

        plt.subplot(3, n, i + 1 + n)
        plt.imshow(lower_half[i].permute(1, 2, 0).numpy())
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(3, n, i + 1 + 2 * n)
        combined = torch.cat((upper_half[i], predictions[i]), dim=1)
        plt.imshow(combined.permute(1, 2, 0).numpy())
        plt.title("Prediction")
        plt.axis('off')

    plt.show()
