# 2025 IEEE Low Power Computer Vision Challenge (CVPR2025) Track 2

This is the repository of the SICer that won the 1st place in CVPR2025-LPCVC-Track2. The team members are  Yuning Ji, Zizhou tong, Zhuohang Li, Xinxin Wang, Chaoyao Shen, Linghui Kong, Chenlong Xia, Bohan Guo, Meng Zhang.

First of all, we would like to thank the LPCVC2025 organizing team for providing such a great opportunity for the participants to practice our skills in edge device model optimization.

This is a fine-tuned version of the Sample Solution from LPCVC 2025 Track 2.

We finetuned the model on 1 A100 GPU that takes about 7 days. 

We considered a variety of resolution solutions and ultimately determined that the accuracy was almost the highest at around 600 resolution（**590** in fact） and the running time was acceptable.The fine-tuned weights have been placed in the folder`./lpcvc_track2_models/model_state_dict.pt`.

We fine-tuned the solution successively twice and found that overfitting would occur if the training time exceeded 20 epochs for this task. Therefore, the weights of several attempts were kept within 20 epochs.



## :fire: News
- [2025.02.13] OpenCV Webinar by Professor Lu introducing LPCVC 2025
- [2025.02.01] Sample solution of Track2: X-Decoder released
- [2025.01.10] LPCVC 2025 accepted as CVPR 2025 Workshop
- [2024.12.10] LPCVC 2025 announced at NeurIPS 2024

## 1. Model Training and Evaluation
:point_right: ***Please refer to [[XDecoder]](https://github.com/microsoft/X-Decoder) for model training and evaluation details.***

### Model Details
- Architecture: Focal-T / ViT-b
- Training data: COCO
- Evaluation data: RefCOCOg
- Task: Grounding Segmentation

### Getting Started
- Training command: `sh command.sh`
- Pre-trained weights: [[Google Drive]](https://drive.google.com/file/d/1pk1HVDvQuGEyGwB4fP6y35mLWqY5xqOq/view?usp=drive_link)
  - Download to: `./lpcvc_track2_models/model_state_dict.pt`

### :bulb: Important Tips
- Higher input resolution generally improves segmentation accuracy but increases computational cost. Consider this trade-off carefully.
- Some complex operations (e.g., GroupNorm, DeformableAttention) may not be well optimized/supported by QNN libraries. Consider using alternative implementations. Check QNN documentation for supported operations.

## 2. Model Compilation and Profiling (Qualcomm AI Hub)
For detailed instructions on model compilation, profiling and inference, refer to the [[AI Hub Documentation]](https://app.aihub.qualcomm.com/docs/).

:point_right: ***LPCVC 2025 Track 2, Sample solution compilation, profiling and inference pipeline available in [[compile_profile_inference_aihub.py]](./compile_and_profile/compile_profile_inference_aihub.py)***

```python
# Submit compilation job to AIHub
compile_job = hub.submit_compile_job(
    model=model_path,
    name="lpcvc25_track2_sample_solution",
    device=hub.Device(deploy_device),
    options="--target_runtime qnn_context_binary",
)
# IMPORTANT! You must share your compile job to lpcvc organizers thus we can pull and evalaute it.
compile_job.modify_sharing(add_emails=['lowpowervision@gmail.com'])
model = compile_job.get_target_model()

# Profile model if requested
profile_job = hub.submit_profile_job(
    name="lpcvc25_track2_sample_solution",
    model=model, 
    device=hub.Device(deploy_device)
)
```

## 3. Inference and Evaluation

:point_right: ***See [[compile_profile_inference_aihub.py]](./compile_and_profile/compile_profile_inference_aihub.py) for complete inference and evaluation pipeline.***

#### :warning: Important Note
During evaluation, only the following inference commands will be used. Ensure your submitted model is correctly compiled and produces valid outputs on AIHub:

```python
# Prepare inputs
aihub_inputs = {
    'image_input': [image_input.detach().cpu().numpy()], 
    'text_input': [text_input.detach().cpu().numpy()]
}

# Run inference
inference_job = hub.submit_inference_job(
    name="lpcvc25_track2_sample_solution",
    model=model,
    device=hub.Device(deploy_device),
    inputs=aihub_inputs
)
qnn_outputs = inference_job.download_output_data() # shape=[1024, 1024], numpy.array
```

### Evaluation Details

#### Device
- Snapdragon X Elite CRD

#### Test Dataset
- 1000 images from ~200 categories
- 3-5 annotated masks per image
- Balanced across mask sizes and categories
- 3-5 text descriptions per mask
- Text descriptions include:
  - Keywords
  - Short phrases
  - Detailed sentences (appearance, location, semantics, relationships, etc.)

#### Input Format
- **Image**:
  - RGB format, shape: 3x1024x1024
  - Longest edge resized to 1024, padded to square
- **Text**:
  - Shape: 2x1x77 (tokenized embedding + attention mask)
  - Uses CLIP tokenizer output

```python
# Image preprocessing example
img = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize(1000, max_size=1024),
])
image = transform(img)
image = torch.from_numpy(np.asanyarray(image)).float().permute(2, 0, 1)
images = [image]
image_input = ImageList.from_tensors(images, size_divisibility=1024).tensor

# All the input images have the same input shape 3x1024x1024 with RGB values [0, 255]. The original images are first resized to make the longest edge equals 1024, then padded to square 1024x1024 by 0s.
```

#### Text Processing
- Uses OpenAI CLIP tokenizer
- Format: `[text_embedding; attention_mask]`

```python
# Text tokenization
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
tokens = tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
text_input = torch.stack((tokens['input_ids'], tokens['attention_mask']))  # Shape: 2x1x77

# (Text tokenization) QNN library does not support tokenization of text input yet. In order to reduce the influence of different text tokenizer used to the final performance, accuracy and latency, we pre-fixed the text tokenizer and only input the tokenized vector of the input text to the model
```

#### Evaluation Metric
- **mIoU** (mean Intersection over Union)

```python
def compute_IoU(pred_seg, gd_seg):
    I = (pred_seg & gd_seg)
    U = (pred_seg | gd_seg)
    return I.sum() / (U.sum() + 1e-6)

# Compute mIoU across test set
pred = output['grounding_mask']  # Binary mask (threshold prediction.sigmoid() > 0.5)
gt = input['groundings']['masks'].bool()
IoUs = [compute_IoU(p, g) for p, g in zip(pred, gt)]
mIoU = sum(IoUs) / len(IoUs) * 100
```



## Acknowledgements
* The sample solution for LPCVC 2025 Track-2 is built on [XDecoder](https://github.com/microsoft/X-Decoder)

## Contact
LPCVC 2025 Organizers:
- Website: [Homepage](https://lpcv.ai/)
- Community: [Slack](https://aihub.qualcomm.com/community/slack)
- Email: [lowpowervision@gmail.com](mailto:lowpowervision@gmail.com)
