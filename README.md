# Capstone-2: Damaged Photo Restoration Using Deep Learning

Capstone 2 repository for Sunway University Computer Science Final year Project.
The scope of this project has been narrowed down to dehazing hazy images, specifically of hazy road conditions.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required dependencies.

```bash
pip install -r requirements.txt
```

## File Structure

To train the model, please ensure that the file structure for the datasets folder is as shown below.

```bash
- datasets/
  - train/
    - main/
      - hazy_image1.png
      - ...
    - hazy/
      - level1/
        - hazy_image1.png
        - hazy_image2.png
        - ...
      - level2/
        - hazy_image1.png
        - hazy_image2.png
        - ...
      - level3/
        - hazy_image1.png
        - hazy_image2.png
        - ...
      - level4/
        - hazy_image1.png
        - hazy_image2.png
        - ...
      - level5/
        - hazy_image1.png
        - hazy_image2.png
        - ...
    - clean/
      - clean_image1.png
      - clean_image2.png
      - ...
  - val/
    - hazy/
      - level1/
        - hazy_image1.png
        - hazy_image2.png
        - ...
      - level2/
        - hazy_image1.png
        - hazy_image2.png
        - ...
      - level3/
        - hazy_image1.png
        - hazy_image2.png
        - ...
      - level4/
        - hazy_image1.png
        - hazy_image2.png
        - ...
      - level5/
        - hazy_image1.png
        - hazy_image2.png
        - ...
    - clean/
      - clean_image1.png
      - clean_image2.png
      - ...


```
