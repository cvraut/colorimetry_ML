**Title:** Extraction and Analysis of Colorimetry Data from Figure Images  

**Objective:**  
The goal of this work was to extract and process long-format data from a colorimetry figure. The extracted data can be used for further analysis and visualization, aiding in the interpretation of experimental results.

**Methods:**  
1. **Image Preprocessing:**  
   - Loaded the image using OpenCV and Matplotlib.
   - Identified unique colors present in the figure.
   - Used KMeans clustering to group similar colors together.

2. **Line Extraction:**  
   - Initial automated attempts to isolate lines using color detection were unsuccessful due to overlapping lines.
   - Manual tracing of individual lines in Photoshop was performed to create separate images for each color.

3. **Data Extraction from Traced Images:**  
   - Processed each traced image to extract x & y coordinates of the colored lines.
   - Applied smoothing techniques (LOWESS) to reduce noise and refine the extracted curves.
   
4. **Visualization and Interpretation:**  
   - Plotted the extracted data to compare with the original figure.
   - Identified key patterns and variations across different colors.

**Key Findings:**  
- Automated extraction was challenging due to line overlap, necessitating manual intervention.
- KMeans clustering helped group similar colors but required fine-tuning.
- The manual tracing approach successfully provided clean datasets for further analysis.
- LOWESS smoothing effectively reduced noise and improved the clarity of extracted data.

**Conclusion:**  
This study developed a semi-automated approach to extract long-format data from a colorimetry image. While automation was limited by image complexity, a combination of manual tracing and computational analysis yielded interpretable results. Future work could explore advanced deep-learning-based segmentation techniques to further automate the process.

