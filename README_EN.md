# InfiniteTalk Multi-Image Digital Human User Guide

## 📖 Table of Contents

1. [Introduction](#introduction)
2. [Quick Reference](#quick-reference)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Node Details](#node-details)
6. [Workflow Examples](#workflow-examples)
7. [FAQ](#faq)
8. [Important Notes](#important-notes)

---

## Quick Reference

### Core Nodes Quick Lookup

| Node Name | Main Function | Key Parameters | Common Use Cases |
|-----------|--------------|----------------|------------------|
| **InfiniteTalkMultiImage** | Multi-image input and timing control | `audio_duration`, `start_time1-20`, `prompt_list_input` | Set start times and prompts for multiple images |
| **InfiniteTalkEmbedsSlice** | Audio embedding slicing | `start_video_frame`, `video_frame_length` | Split long audio into segments for processing |
| **VideoFromPathsAndAudio** | Video synthesis | `image_paths`, `frame_counts`, `transition_type` | Combine image sequences into video |
| **MakeBatchFromIntList** | Integer list merging | `int_list` | Merge multiple integer values |
| **GetIntByIndex** | Get integer by index | `int_batch`, `index` | Extract value at specified index from batch |
| **MakeBatchFromFloatList** | Float list merging | `float_list` | Merge multiple float values |
| **GetFloatByIndex** | Get float by index | `float_batch`, `index` | Extract value at specified index from batch |

### Transition Effects Quick Lookup

| Effect Name | Visual Effect | Use Cases |
|-------------|--------------|-----------|
| `crossfade` | Fade in/out | General purpose, most natural |
| `left_wipe` | Wipe from left to right | Emphasize direction |
| `right_wipe` | Wipe from right to left | Emphasize direction |
| `slide_left` | Slide left | Strong dynamic feel |
| `slide_right` | Slide right | Strong dynamic feel |
| `zoom_in` | Zoom in | Emphasize focus |
| `zoom_out` | Zoom out | Exit effect |
| `soft_blur` | Blur transition | Soft transition |
| `random` | Random effect | Variety |

---

## Introduction

**InfiniteTalk Multi-Image Digital Human** is a ComfyUI plugin that allows you to:
- ✅ Create digital human videos using multiple images
- ✅ Specify different start times for each image
- ✅ Set independent prompts (Prompts) for each image
- ✅ Generate infinitely long videos without increasing VRAM usage as video duration increases
- ✅ Support multiple image transition effects (fade, slide, zoom, etc.)
- ✅ Fully compatible with KJ nodes without modifying existing nodes

---

## Installation

### Prerequisites

1. **ComfyUI Installed**
   - Ensure your ComfyUI is running properly

2. **Install Dependencies**
   ```bash
   # Navigate to plugin directory
   cd ComfyUI-InfiniteTalk-MultiImage
   
   # Install Python dependencies
   pip install -r requirements.txt
   ```

3. **Install FFmpeg** (for video synthesis)
   - Windows: Download [FFmpeg](https://ffmpeg.org/download.html) and add to system PATH
   - Verify installation: Run `ffmpeg -version` in command line

### Install Plugin

1. Copy the `ComfyUI-InfiniteTalk-MultiImage` folder to ComfyUI's `custom_nodes` directory
2. Restart ComfyUI
3. Look for the `InfiniteTalk` category in the node menu to confirm successful installation

---

## Quick Start

### Basic Workflow

1. **Prepare Materials**
   - Prepare one or more person images (recommended: same resolution)
   - Prepare an audio file (.wav format)

2. **Load Workflow**
   - Open ComfyUI
   - Load the `InfiniteTalk 图片数字人--多图.json` workflow file

3. **Set Parameters**
   - In the `InfiniteTalkMultiImage` node:
     - Enter audio duration (seconds)
     - Connect multiple images
     - Set start time for each image
     - Enter prompts (one per line)

4. **Run Generation**
   - Click "Queue Prompt" to start generation
   - After completion, check the generated video in the output directory

### Parameter Setting Tips

#### Time Setting Recommendations

**Equal Duration Distribution** (Recommended for beginners):
```
Audio Duration: 9.0 seconds
Image 1: 0.0 seconds
Image 2: 3.0 seconds
Image 3: 6.0 seconds
```
Each image displays for 3 seconds, simple and clear.

**Unequal Duration Distribution**:
```
Audio Duration: 10.0 seconds
Image 1: 0.0 seconds  (displays 4 seconds)
Image 2: 4.0 seconds  (displays 3 seconds)
Image 3: 7.0 seconds  (displays 3 seconds)
```
You can allocate duration based on content importance.

#### Prompt Writing Tips

**Format**: One prompt per line, corresponding to one image

**Example**:
```
Front view, smile, natural expression
Look at camera, blink, slight nod
Wave hand, greet, keep smiling
```

**Tips**:
- Describe specific actions and expressions
- Keep it concise, avoid being too long
- If an image doesn't need special control, you can leave it empty

#### Transition Effect Selection Tips

- **Daily use**: `crossfade` (fade in/out, most natural)
- **Need dynamic feel**: `slide_left` or `slide_right`
- **Emphasize focus**: `zoom_in`
- **Want variety**: `random` (randomly selects each time)

---

## Node Details

### 1. InfiniteTalkMultiImage (Core Node)

**Function**: Processes multiple image inputs and calculates frame count and start time for each image.

#### Input Parameters

| Parameter Name | Type | Description | Default Value |
|----------------|------|-------------|---------------|
| `audio_duration` | FLOAT | Total audio duration (seconds) | 8.0 |
| `prompt_list_input` | STRING | Prompt list, one per line, corresponding to each image | "" |
| `image1` ~ `image20` | IMAGE | Supports up to 20 image inputs | - |
| `start_time1` ~ `start_time20` | FLOAT | Start time for each image (seconds), step 0.04 | 0.0, 3.0, 6.0... |
| `auto_start_time_list` | START_TIME_LIST | Auto time list (optional, overrides manually set times) | [] |

#### Output Parameters

| Output Name | Type | Description |
|-------------|------|-------------|
| `image_list` | IMAGE | Processed image list |
| `frame_count_list` | INT | Frame count list for each image |
| `real_start_time_list` | FLOAT | Actually used start time list |
| `prompt_list` | STRING | Aligned prompt list |

#### Usage Example

```
Audio Duration: 10.0 seconds
Image 1: Start time 0.0 seconds
Image 2: Start time 3.0 seconds
Image 3: Start time 6.0 seconds

Prompts:
Smile, natural expression
Look at camera, blink
Wave hand, greet
```

**How It Works**:
- Automatically calculates the number of frames needed for each image based on audio duration and each image's start time
- If an image's start time exceeds audio duration, that image will be automatically filtered
- Prompts are automatically aligned to image count (insufficient prompts filled with empty strings, excess prompts truncated)

---

### 2. InfiniteTalkEmbedsSlice (Embedding Slice Node)

**Function**: Slices audio embeddings (embeds) by video frame intervals.

#### Input Parameters

| Parameter Name | Type | Description | Default Value |
|----------------|------|-------------|---------------|
| `multitalk_embeds` | MULTITALK_EMBEDS | Audio embedding data | - |
| `start_video_frame` | INT | Video start frame number | 0 |
| `video_frame_length` | INT | Video frame length | 30 |
| `fps` | INT | Video frame rate (1-120) | 25 |

#### Output Parameters

| Output Name | Type | Description |
|-------------|------|-------------|
| `sliced_multitalk_embeds` | MULTITALK_EMBEDS | Sliced audio embeddings |

#### Usage Notes

- This node is used to split long audio embedding data into multiple segments
- Audio embedding frame rate is fixed at 25 frames/second
- The node automatically converts video frame intervals to corresponding audio frame intervals

**Example**:
```
Video frames: 0-30 (30 frames)
Video frame rate: 25 fps
Audio frame rate: 25 fps

Result: Audio embedding frames 0-30
```

---

### 3. VideoFromPathsAndAudio (Video Synthesis Node)

**Function**: Combines image sequences and audio into video, supports multiple transition effects.

#### Input Parameters

| Parameter Name | Type | Description | Default Value |
|----------------|------|-------------|---------------|
| `image_paths` | STRING | Image path list, one per line (supports folders or files) | "" |
| `frame_counts` | STRING | Frame count for each corresponding path | "" |
| `audio` | AUDIO | Audio input | - |
| `fps` | INT | Output video frame rate (1-120) | 25 |
| `filename_prefix` | STRING | Output filename prefix | "video_output" |
| `format` | SELECT | Output format: mp4/mkv/mov | "mp4" |
| `enable_transition` | BOOLEAN | Enable transition effects | True |
| `transition_seconds` | FLOAT | Transition duration (seconds, 0.05-5.0) | 0.5 |
| `transition_type` | SELECT | Transition type (see below) | "crossfade" |

#### Transition Type Descriptions

| Type | Effect Description |
|------|-------------------|
| `crossfade` | Fade in/out (most commonly used, natural and smooth) |
| `left_wipe` | Wipe from left to right |
| `right_wipe` | Wipe from right to left |
| `slide_left` | Slide left (new image slides in from right) |
| `slide_right` | Slide right (new image slides in from left) |
| `zoom_in` | Zoom in (new image zooms from 70% to 100%) |
| `zoom_out` | Zoom out (old image shrinks, new image appears) |
| `soft_blur` | Blur transition (old image gradually blurs, new image becomes clear) |
| `random` | Randomly select transition effect |

#### Output Parameters

| Output Name | Type | Description |
|-------------|------|-------------|
| `video_path` | STRING | Generated video file path |

#### Usage Examples

**Image Path Format**:
```
D:\images\frame_001.png
D:\images\frame_002.png
D:\images\frame_003.png
```

**Frame Count Format** (one per line, corresponding to each path):
```
10
15
20
```

**Folder Support**:
```
D:\images\folder1
D:\images\folder2
```

The node automatically reads all images in the folder (supports png, jpg, jpeg, bmp, webp).

---

### 4. MakeBatchFromIntList (Integer List to Batch)

**Function**: Merges multiple integer inputs into a batch.

#### Input Parameters

| Parameter Name | Type | Description |
|----------------|------|-------------|
| `int_list` | INT | Integer list (supports multiple inputs) |

#### Output Parameters

| Output Name | Type | Description |
|-------------|------|-------------|
| `int_batch` | INT | Merged integer batch |

#### Usage Notes

- Used to merge multiple individual integer values into a batch
- If there's only one input, returns it directly
- Multiple inputs are converted to PyTorch tensor

---

### 5. GetIntByIndex (Get Integer by Index)

**Function**: Gets the value at specified index from an integer batch.

#### Input Parameters

| Parameter Name | Type | Description | Default Value |
|----------------|------|-------------|---------------|
| `int_batch` | INT | Integer batch | - |
| `index` | INT | Index to retrieve (starts from 0) | 0 |

#### Output Parameters

| Output Name | Type | Description |
|-------------|------|-------------|
| `value` | INT | Integer value at specified index |

#### Usage Notes

- If index is out of range, returns boundary value (first or last)
- If batch is empty, returns 0

---

### 6. MakeBatchFromFloatList (Float List to Batch)

**Function**: Merges multiple float inputs into a batch.

#### Input Parameters

| Parameter Name | Type | Description |
|----------------|------|-------------|
| `float_list` | FLOAT | Float list (supports multiple inputs) |

#### Output Parameters

| Output Name | Type | Description |
|-------------|------|-------------|
| `float_batch` | FLOAT | Merged float batch |

#### Usage Notes

- Similar to `MakeBatchFromIntList`, but handles floats
- Used to merge multiple float values

---

### 7. GetFloatByIndex (Get Float by Index)

**Function**: Gets the value at specified index from a float batch.

#### Input Parameters

| Parameter Name | Type | Description | Default Value |
|----------------|------|-------------|---------------|
| `float_batch` | FLOAT | Float batch | - |
| `index` | INT | Index to retrieve (starts from 0) | 0 |

#### Output Parameters

| Output Name | Type | Description |
|-------------|------|-------------|
| `value` | FLOAT | Float value at specified index |

#### Usage Notes

- If index is out of range, returns boundary value
- If batch is empty, returns 0.0

---

## Workflow Examples

### Example 1: Three Images, 3 Seconds Each

**Settings**:
- Audio Duration: 9.0 seconds
- Image 1: Start time 0.0 seconds
- Image 2: Start time 3.0 seconds
- Image 3: Start time 6.0 seconds

**Prompts**:
```
Smile, natural expression
Look at camera, blink
Wave hand, greet
```

**Result**: Each image displays for 3 seconds, total 9-second video.

---

### Example 2: Two Images, Different Durations

**Settings**:
- Audio Duration: 10.0 seconds
- Image 1: Start time 0.0 seconds (displays 5 seconds)
- Image 2: Start time 5.0 seconds (displays 5 seconds)

**Prompts**:
```
Front view, smile
Side view, turn head
```

**Result**: First image displays for 5 seconds, second image displays for 5 seconds.

---

### Example 3: Using Transition Effects

**Settings**:
- Enable Transition: ✅
- Transition Duration: 0.5 seconds
- Transition Type: `crossfade` (fade in/out)

**Effect**: Images will have a smooth 0.5-second transition when switching, no abrupt changes.

---

## FAQ

### Q1: Why is my image not being used?

**Possible Reasons**:
1. The image's start time exceeds the audio duration
2. The image is not properly connected to the node

**Solutions**:
- Check if `audio_duration` is greater than all image start times
- Ensure image nodes are properly connected to the `InfiniteTalkMultiImage` node

---

### Q2: What if I don't have enough prompts?

**Note**: If the number of prompts is less than the number of images, insufficient parts will be filled with empty strings.

**Recommendation**: Write a prompt for each image to get better control.

---

### Q3: Transition effects not working?

**Checklist**:
1. Is `enable_transition` set to `True`?
2. Are there multiple image segments? (Single image cannot transition)
3. Is the transition duration reasonable? (Recommended 0.3-1.0 seconds)

---

### Q4: Video generation failed with FFmpeg error?

**Solutions**:
1. Confirm FFmpeg is properly installed and added to PATH
2. Run `ffmpeg -version` in command line to verify
3. Check if output directory has write permissions

---

### Q5: Out of VRAM?

**Recommendations**:
1. Use the `InfiniteTalk 图片数字人--多图--低显存16G版本.json` workflow
2. Reduce the number of images processed simultaneously
3. Lower image resolution
4. Use smaller batch sizes

---

### Q6: Time Precision Issues

**Note**: 
- Minimum step size for start time is 0.04 seconds (corresponds to 1 frame at 25 fps)
- Input times are automatically aligned to multiples of 0.04

**Examples**:
- Input 3.05 seconds → automatically aligned to 3.04 seconds
- Input 3.07 seconds → automatically aligned to 3.08 seconds

---

## Important Notes

### ⚠️ Important Tips

1. **Time Settings**
   - Start time must be less than audio duration
   - Recommend arranging images in chronological order (0s → 3s → 6s...)

2. **Image Format**
   - Supports common image formats: PNG, JPG, JPEG, BMP, WEBP
   - Recommend all images have the same resolution to avoid size issues

3. **Audio Format**
   - Recommend using WAV format
   - Ensure audio duration is accurate

4. **Transition Effects**
   - Don't set transition duration too long (recommended 0.3-1.0 seconds)
   - Transitions consume image display time, calculate accordingly

5. **Performance Optimization**
   - If VRAM is insufficient, use the low-VRAM version workflow
   - For large batch processing, recommend generating in batches

6. **File Paths**
   - Windows paths can use backslash `\` or forward slash `/`
   - When paths contain spaces, recommend wrapping in quotes

---

## Technical Support

If you encounter problems:
1. Check ComfyUI console for error messages
2. Confirm all dependencies are properly installed
3. Review workflow example files and compare settings

---

## Changelog

- **v1.0**: Initial release
  - Support multi-image input
  - Support time control
  - Support prompt control
  - Support multiple transition effects

---

**Enjoy using!** 🎉

