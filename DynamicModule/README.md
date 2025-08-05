# Dynamic Module - Multi-Modal Video Generation

A comprehensive tool for dynamic video generation with multiple modules.

## Requirements

- Python 3.8+
- CUDA 11.0+ (recommended)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download pre-trained models to pretrained_models/ directory
```

## Usage

### Basic Usage

```bash
python main.py --input_image path/to/image.jpg --prompt "description text" --model_name "model_name"
```

### Examples
```python
 python main.py     --input_image demo/pexels/1.png     --prompt "A turtle sunbathes on a mossy rock by a river."     --model_name "Dynamic_Module"     --camera_pose_type "zoom in"     --trace_extract_ratio 0.1     --trace_scale_factor 1.0     --frame_stride 2     --steps 25     --camera_cfg 1.0     --cfg_scale 5.5     --seed 12333     --enable_camera_condition     --device cuda
```

### Available Models

**DynamicModule (Camera-Controlled Image-to-Video):**
- Converts static images to dynamic videos with camera control
- Supports various camera pose types for different motion effects

### Parameters

**Required Parameters:**
- `--input_image`: Path to input image
- `--prompt`: Text description for video generation
- `--model_name`: Name of the model to use

**Optional Parameters:**
- `--camera_pose_type`: Camera pose type (default: stationary)
- `--negative_prompt`: Negative prompt
- `--steps`: Sampling steps (default: 25)
- `--cfg_scale`: CFG scale (default: 5.5)
- `--seed`: Random seed (default: 12333)
- `--use_qwen2vl_captioner`: Use automatic caption generation
- `--device`: Device to use (default: cuda)
- `--result_dir`: Directory to save results (default: ./demo/results)

## Output

Generated video files are saved in the `./demo/results/` directory.
