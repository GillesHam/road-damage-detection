# RDD2022 China Drone 300 Raw Dataset

This folder contains the same 300 selected RDD2022 China Drone images, without a train/validation/test split.

Layout:

- `images`: 300 JPG images in one flat folder.
- `annotations/xmls`: 300 matching Pascal VOC XML annotation files.
- `source_manifest.json`: records which previous split each file came from.

Source dataset:

https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547

License: CC BY 4.0.

Classes kept:

- `D00`: longitudinal crack.
- `D10`: transverse crack.
- `D20`: alligator crack.
- `D40`: pothole.

Extra labels such as `Repair` and `Block crack` were excluded.
