# Transforming Precision Agriculture through Deep Learning: CNN Encoder- Decoders and a Hybrid YOLOv8-SAM Solution for Advanced Grape Cluster Seg- mentation
Image segmentation using the GrapesNet dataset.

## GrapesNet Dataset
The GrapesNet dataset consists of four distinct grape bunch datasets from vineyards, detailed below:
1. **Dataset-1**: RGB images featuring a single grape bunch against a black background and another set with a natural background.
2. **Dataset-2**: RGB images showcasing two or more grape bunches within a natural background.
3. **Dataset-3**: RGB-D images capturing vineyard scenes under real environmental conditions.
4. **Dataset-4**: RGB-D images containing one grape bunch set in an experimental environment against a coral background for grape weight prediction tasks.

**Dataset link**  
[Link to GrapesNet Dataset](https://data.mendeley.com/datasets/mhzmzd5cwx/1)

## Models

### SegNet

SegNet is a deep learning framework specifically developed for semantic segmentation in computer vision. It enables pixel-wise classification in images, allowing each pixel to be assigned a class label for identifying objects or regions. Key features include:

1. **Encoder-Decoder Structure**: SegNet utilizes an encoder-decoder design, where the encoder extracts features and the decoder reconstructs the segmented output.
  
2. **Convolutional Neural Networks (CNNs)**: Both components consist of multiple convolutional layers. The encoder reduces the spatial dimensions while extracting hierarchical features, and the decoder upsamples the low-resolution feature maps to create a high-resolution segmentation map.

3. **Pooling Indices**: A unique aspect of SegNet is its use of max-pooling indices during downsampling in the encoder. Instead of simply storing maximum values, it keeps the indices of these values for precise pixel-wise classification in the decoder.

4. **Skip Connections**: SegNet incorporates skip connections linking corresponding layers in the encoder and decoder to enhance segmentation accuracy by preserving fine-grained details.

5. **Softmax Layer**: The final layer of the decoder generally includes a softmax activation function to produce class probability scores for each pixel, assigning the class with the highest score.

6. **Applications**: SegNet has applications in various fields, including autonomous driving, medical image analysis, and robotics.

7. **Advantages**: SegNet’s architecture allows for efficient pixel-wise classification, making it suitable for real-time applications while maintaining spatial information through its unique features.

8. **Limitations**: Despite its strengths, SegNet may struggle with objects of varying sizes and complex scenes. More advanced architectures, such as U-Net and DeepLab, have been developed to address these challenges.

In conclusion, SegNet is an encoder-decoder architecture focused on semantic segmentation, leveraging max-pooling indices and skip connections to maintain spatial details. 

## SegNet Results
### Input Image
![Input Image](https://github.com/Ni-vik/ImageSegmentation/blob/main/images/input.png)

### Output Image
![Output Image](https://github.com/Ni-vik/ImageSegmentation/blob/main/images/segnet.png)

## ResNet50
A ResNet-50 based Pyramid Scene Parsing Network (PSPNet) is a sophisticated deep learning architecture for image segmentation, merging the strengths of ResNet-50 for feature extraction with PSPNet for scene parsing. Highlights include:

1. **ResNet-50 Backbone**: Utilizes ResNet-50 for effective feature extraction while mitigating the vanishing gradient issue. 

2. **Feature Extraction**: The backbone processes the input image, capturing hierarchical features at varying scales.

3. **PSPNet Module**: This module, added atop ResNet-50, captures contextual information at multiple scales through spatial pyramid pooling, enhancing the understanding of the global context for accurate segmentation.

4. **Semantic Segmentation**: The final layers focus on semantic segmentation, leveraging features from ResNet-50 and context from PSPNet to create pixel-wise segmentation maps.

5. **Applications**: Widely utilized in tasks requiring precise semantic segmentation, such as autonomous driving and medical imaging.

6. **Advantages**: The combination of ResNet-50 and PSPNet enables robust image segmentation capabilities, effective in complex scenes.

7. **Challenges**: Training deep networks like this can be computationally intensive and demand significant GPU resources along with adequate labeled data.

In summary, a ResNet-50 based PSPNet leverages feature extraction and contextual information for effective semantic segmentation in diverse applications.

## Results using ResNet50 and PSPNet
### Input Image
![Input Image](https://github.com/Ni-vik/ImageSegmentation/blob/main/images/input.png)

### Output Image
![Output Image](https://github.com/Ni-vik/ImageSegmentation/blob/main/images/resnet%2050%20segmented.png)

## VGG_UNet
### Results using VGG_UNet
### Input Image
![Input Image](https://github.com/Ni-vik/ImageSegmentation/blob/main/images/input.png)

### Output Image
![Output Image](https://github.com/Ni-vik/ImageSegmentation/blob/main/images/vgg_unet.png)

## Detectron
**Backbone CNN**: Mask R-CNN employs a backbone CNN, such as ResNet or VGG, to extract vital features from input images. 

1. **Region Proposal Network (RPN)**: Generates potential bounding boxes that might contain objects, evaluating the likelihood of each region.

2. **RoI Align**: This method extracts fixed-size feature maps, ensuring alignment with object boundaries for spatial accuracy.

3. **Object Classification Head**: Predicts class labels for each proposed region and assesses confidence in these predictions.

4. **Bounding Box Regression Head**: Refines bounding boxes for better accuracy by adjusting sizes and positions.

5. **Mask Generation Head**: Unique to Mask R-CNN, this component predicts pixel-level masks for each object, outlining their shapes.

6. **Combination and Output**: Outputs include the class label, refined bounding box, and detailed pixel-level mask for each object.

## Results using Detectron-2
![Detection Results](https://github.com/Ni-vik/ImageSegmentation/blob/main/images/WhatsApp%20Image%202023-09-20%20at%2015.27.04.jpg)

## FCN
The Fully Convolutional Network (FCN) is designed for pixel-level segmentation. It utilizes a feature extractor like VGG or ResNet, replacing fully connected layers with convolutional layers to preserve spatial information. Upsampling layers enhance feature map resolution, while skip connections integrate high and low-level features for improved accuracy. The final convolutional layer generates pixel-wise class predictions.

## Results using FCN
![FCN Results](https://github.com/Ni-vik/ImageSegmentation/blob/main/images/fcn.jpg)

### Performing Image Segmentation on Your Own Dataset
To segment images in your dataset, you will need annotated images.

### Obtaining Annotated Images
There are two methods to achieve this:
1. **Edge Detection Techniques**
2. **Manual Annotation**: Use annotation tools like the VGG Image Annotator (VIA).
