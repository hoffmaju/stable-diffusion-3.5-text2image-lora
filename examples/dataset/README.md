# Example Dataset

This folder contains an example dataset structure for SD3.5 LoRA training. This is a **template** to help you understand the required format.

## 📁 Structure

```
examples/dataset/
├── metadata.jsonl          # Dataset metadata with image paths and captions
├── images/                 # Directory containing training images
│   ├── landscape_001.jpg   # Example image files (you need to add actual images)
│   ├── portrait_001.jpg
│   ├── cityscape_001.jpg
│   └── ...                # More images as referenced in metadata.jsonl
└── README.md              # This file
```

## 📝 Metadata Format

The `metadata.jsonl` file contains one JSON object per line, each with:

-   `image`: Relative path to the image file (from dataset root)
-   `caption`: Descriptive text for the image

Example:

```json
{"image": "images/landscape_001.jpg", "caption": "a breathtaking mountain landscape with snow-capped peaks"}
{"image": "images/portrait_001.jpg", "caption": "a professional portrait of a young woman with curly brown hair"}
```

## 🖼️ Images

The `images/` folder should contain your actual training images. The current `metadata.jsonl` references 15 example images that you need to provide:

-   landscape_001.jpg
-   portrait_001.jpg
-   cityscape_001.jpg
-   nature_001.jpg
-   abstract_001.jpg
-   animal_001.jpg
-   architecture_001.jpg
-   fantasy_001.jpg
-   food_001.jpg
-   space_001.jpg
-   vintage_001.jpg
-   underwater_001.jpg
-   winter_001.jpg
-   desert_001.jpg
-   city_night_001.jpg

## 🔄 Using Your Own Data

To use your own dataset:

1. **Replace images**: Add your training images to the `images/` folder
2. **Update metadata**: Modify `metadata.jsonl` to reference your images with appropriate captions
3. **Ensure paths match**: Make sure image paths in metadata.jsonl match your actual files

### Tips for Good Training Data:

-   **Image Quality**: Use high-resolution images (1024x1024 recommended)
-   **Diverse Content**: Include variety in subjects, styles, and compositions
-   **Accurate Captions**: Write detailed, descriptive captions
-   **Consistent Naming**: Use clear, systematic file names
-   **Sufficient Quantity**: At least 50-100 images for good results

## 🚀 Training

Once you have your images in place, you can start training:

```bash
# Basic training
bash scripts/train_basic.sh

# Advanced training
bash scripts/train_advanced.sh
```

## 📋 Data Collection Tips

Consider these sources for training data:

-   Personal photography collections
-   Creative Commons licensed images
-   Generated images from base SD3.5 (for style transfer)
-   Specific domain datasets (art, photos, etc.)

**Important**: Ensure you have rights to use all training images!
