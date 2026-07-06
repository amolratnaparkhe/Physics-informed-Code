import numpy as np
import pydicom
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.patches import Polygon
import cv2
import glob
import os

class TumorVoxelExtractor:
    def __init__(self, cbct_folder, rtstruct_file):
        """
        Initialize tumor voxel extractor
        
        Parameters:
        - cbct_folder: Path to folder containing CBCT DICOM files
        - rtstruct_file: Path to RTSTRUCT DICOM file
        """
        self.cbct_folder = cbct_folder
        self.rtstruct_file = rtstruct_file
        
        # Load data
        self.cbct_slices = self.load_cbct_series()
        self.rtstruct = pydicom.dcmread(rtstruct_file)
        
        # Create 3D volume and coordinate system
        self.cbct_volume, self.voxel_spacing, self.origin = self.create_3d_volume()
        
        print(f"Loaded {len(self.cbct_slices)} CBCT slices")
        print(f"Volume shape: {self.cbct_volume.shape}")
        print(f"Voxel spacing: {self.voxel_spacing} mm")
        print(f"Origin: {self.origin} mm")
    
    def load_cbct_series(self):
        """Load and sort CBCT DICOM series"""
        cbct_files = glob.glob(os.path.join(self.cbct_folder, "*.dcm"))
        
        dicom_slices = []
        for file_path in cbct_files:
            try:
                ds = pydicom.dcmread(file_path)
                if hasattr(ds, 'pixel_array') and hasattr(ds, 'ImagePositionPatient'):
                    dicom_slices.append(ds)
            except:
                continue
        
        # Sort by Z position (Image Position Patient)
        dicom_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        
        return dicom_slices
    
    def create_3d_volume(self):
        """Create 3D volume from CBCT slices"""
        if not self.cbct_slices:
            raise ValueError("No CBCT slices loaded")
        
        # Get dimensions
        rows = self.cbct_slices[0].Rows
        cols = self.cbct_slices[0].Columns
        num_slices = len(self.cbct_slices)
        
        # Create 3D array
        volume = np.zeros((num_slices, rows, cols), dtype=np.float32)
        
        # Fill the volume
        for i, ds in enumerate(self.cbct_slices):
            volume[i] = ds.pixel_array.astype(np.float32)
        
        # Get coordinate system information
        pixel_spacing = self.cbct_slices[0].PixelSpacing  # [row, col]
        slice_thickness = abs(float(self.cbct_slices[1].ImagePositionPatient[2]) - 
                             float(self.cbct_slices[0].ImagePositionPatient[2]))
        
        voxel_spacing = [slice_thickness, pixel_spacing[0], pixel_spacing[1]]  # [z, y, x]
        origin = self.cbct_slices[0].ImagePositionPatient  # [x, y, z]
        
        return volume, voxel_spacing, origin
    
    def get_roi_contours(self, roi_name="CTV"):
        """Extract contour data for specific ROI"""
        # Find the ROI number
        roi_number = None
        for roi in self.rtstruct.StructureSetROISequence:
            if roi_name.upper() in roi.ROIName.upper():
                roi_number = roi.ROINumber
                print(f"Found {roi.ROIName} with ROI Number: {roi_number}")
                break
        
        if roi_number is None:
            available_rois = [roi.ROIName for roi in self.rtstruct.StructureSetROISequence]
            raise ValueError(f"ROI '{roi_name}' not found. Available ROIs: {available_rois}")
        
        # Extract contours
        contours_by_slice = {}
        
        for roi_contour in self.rtstruct.ROIContourSequence:
            if roi_contour.ReferencedROINumber == roi_number:
                for contour in roi_contour.ContourSequence:
                    # Get contour data (x, y, z coordinates)
                    contour_data = np.array(contour.ContourData).reshape(-1, 3)
                    z_coord = contour_data[0, 2]  # Z coordinate
                    
                    if z_coord not in contours_by_slice:
                        contours_by_slice[z_coord] = []
                    
                    contours_by_slice[z_coord].append(contour_data)
        
        return contours_by_slice
    
    def world_to_voxel_coords(self, world_coords):
        """Convert world coordinates to voxel indices"""
        # world_coords: [x, y, z] in mm
        # Returns: [slice_idx, row_idx, col_idx]
        
        x, y, z = world_coords
        origin_x, origin_y, origin_z = self.origin
        
        # Convert to voxel coordinates
        col_idx = (x - origin_x) / self.voxel_spacing[2]  # x direction
        row_idx = (y - origin_y) / self.voxel_spacing[1]  # y direction
        
        # Find closest slice
        slice_positions = [float(ds.ImagePositionPatient[2]) for ds in self.cbct_slices]
        slice_idx = np.argmin(np.abs(np.array(slice_positions) - z))
        
        return slice_idx, int(row_idx), int(col_idx)
    
    def create_tumor_mask(self, roi_name="CTV"):
        """Create 3D binary mask for the tumor"""
        contours_by_slice = self.get_roi_contours(roi_name)
        
        # Initialize mask
        mask = np.zeros_like(self.cbct_volume, dtype=bool)
        
        print(f"Creating mask for {len(contours_by_slice)} slices with contours...")
        
        for z_world, contour_list in contours_by_slice.items():
            # Find corresponding slice index
            slice_positions = [float(ds.ImagePositionPatient[2]) for ds in self.cbct_slices]
            slice_idx = np.argmin(np.abs(np.array(slice_positions) - z_world))
            
            # Create mask for this slice
            slice_mask = np.zeros((self.cbct_volume.shape[1], self.cbct_volume.shape[2]), dtype=bool)
            
            for contour_data in contour_list:
                # Convert contour points to pixel coordinates
                pixel_points = []
                for point in contour_data:
                    _, row, col = self.world_to_voxel_coords(point)
                    pixel_points.append([col, row])  # OpenCV uses (x, y) format
                
                pixel_points = np.array(pixel_points, dtype=np.int32)
                
                # Create filled polygon
                temp_mask = np.zeros_like(slice_mask, dtype=np.uint8)
                cv2.fillPoly(temp_mask, [pixel_points], 1)
                slice_mask = slice_mask | (temp_mask > 0)
            
            mask[slice_idx] = slice_mask
        
        return mask
    
    def extract_tumor_voxels(self, roi_name="CTV"):
        """Extract tumor voxel values and coordinates"""
        # Create tumor mask
        tumor_mask = self.create_tumor_mask(roi_name)
        
        # Extract voxel values
        tumor_voxels = self.cbct_volume[tumor_mask]
        
        # Extract voxel coordinates (in voxel space)
        tumor_coords_voxel = np.argwhere(tumor_mask)  # Returns [slice, row, col]
        
        # Convert to world coordinates
        tumor_coords_world = []
        for slice_idx, row_idx, col_idx in tumor_coords_voxel:
            # Convert voxel indices to world coordinates
            x = self.origin[0] + col_idx * self.voxel_spacing[2]
            y = self.origin[1] + row_idx * self.voxel_spacing[1]
            z = float(self.cbct_slices[slice_idx].ImagePositionPatient[2])
            
            tumor_coords_world.append([x, y, z])
        
        tumor_coords_world = np.array(tumor_coords_world)
        
        print(f"Extracted {len(tumor_voxels)} tumor voxels")
        print(f"Tumor volume: {len(tumor_voxels) * np.prod(self.voxel_spacing):.2f} mm³")
        
        return tumor_voxels, tumor_coords_voxel, tumor_coords_world, tumor_mask
    
    def calculate_center_of_mass(self, roi_name="CTV", method='geometric'):
        """
        Calculate center of mass of the tumor
        
        Parameters:
        - method: 'geometric' (unweighted) or 'intensity' (weighted by voxel values)
        """
        tumor_voxels, tumor_coords_voxel, tumor_coords_world, tumor_mask = self.extract_tumor_voxels(roi_name)
        
        if method == 'geometric':
            # Geometric center of mass (unweighted)
            com_world = np.mean(tumor_coords_world, axis=0)
            com_voxel = np.mean(tumor_coords_voxel, axis=0)
            
        elif method == 'intensity':
            # Intensity-weighted center of mass
            # Normalize intensities to positive values
            normalized_intensities = tumor_voxels - np.min(tumor_voxels) + 1
            
            # Calculate weighted center of mass
            total_weight = np.sum(normalized_intensities)
            com_world = np.sum(tumor_coords_world * normalized_intensities[:, np.newaxis], axis=0) / total_weight
            com_voxel = np.sum(tumor_coords_voxel * normalized_intensities[:, np.newaxis], axis=0) / total_weight
            
        else:
            raise ValueError("Method must be 'geometric' or 'intensity'")
        
        print(f"\n=== Center of Mass ({method}) ===")
        print(f"World coordinates: ({com_world[0]:.2f}, {com_world[1]:.2f}, {com_world[2]:.2f}) mm")
        print(f"Voxel coordinates: ({com_voxel[0]:.1f}, {com_voxel[1]:.1f}, {com_voxel[2]:.1f})")
        
        return com_world, com_voxel, tumor_mask
    
    def visualize_tumor_and_com(self, roi_name="CTV", method='geometric'):
        """Visualize tumor with center of mass"""
        com_world, com_voxel, tumor_mask = self.calculate_center_of_mass(roi_name, method)
        
        # Find slice containing center of mass
        com_slice = int(com_voxel[0])
        com_row = int(com_voxel[1])
        com_col = int(com_voxel[2])
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Axial view (at COM slice)
        axial_img = self.cbct_volume[com_slice]
        axial_mask = tumor_mask[com_slice]
        
        axes[0, 0].imshow(axial_img, cmap='gray', aspect='equal')
        axes[0, 0].contour(axial_mask, colors='red', linewidths=2)
        axes[0, 0].plot(com_col, com_row, 'r+', markersize=15, markeredgewidth=3)
        axes[0, 0].set_title(f'Axial - Slice {com_slice}')
        axes[0, 0].axis('off')
        
        # Sagittal view (at COM column)
        sagittal_img = self.cbct_volume[:, :, com_col].T
        sagittal_mask = tumor_mask[:, :, com_col].T
        
        axes[0, 1].imshow(sagittal_img, cmap='gray', aspect='equal')
        axes[0, 1].contour(sagittal_mask, colors='red', linewidths=2)
        axes[0, 1].plot(com_slice, com_row, 'r+', markersize=15, markeredgewidth=3)
        axes[0, 1].set_title(f'Sagittal - Column {com_col}')
        axes[0, 1].axis('off')
        
        # Coronal view (at COM row)
        coronal_img = self.cbct_volume[:, com_row, :]
        coronal_mask = tumor_mask[:, com_row, :]
        
        axes[0, 2].imshow(coronal_img, cmap='gray', aspect='equal')
        axes[0, 2].contour(coronal_mask, colors='red', linewidths=2)
        axes[0, 2].plot(com_col, com_slice, 'r+', markersize=15, markeredgewidth=3)
        axes[0, 2].set_title(f'Coronal - Row {com_row}')
        axes[0, 2].axis('off')
        
        # 3D visualization of tumor extent
        tumor_coords = np.argwhere(tumor_mask)
        
        axes[1, 0].scatter(tumor_coords[:, 2], tumor_coords[:, 1], 
                          c=tumor_coords[:, 0], cmap='viridis', s=1, alpha=0.6)
        axes[1, 0].plot(com_col, com_row, 'r+', markersize=15, markeredgewidth=3)
        axes[1, 0].set_xlabel('Column (X)')
        axes[1, 0].set_ylabel('Row (Y)')
        axes[1, 0].set_title('Tumor Extent (X-Y view)')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].scatter(tumor_coords[:, 0], tumor_coords[:, 1], 
                          c=tumor_coords[:, 2], cmap='viridis', s=1, alpha=0.6)
        axes[1, 1].plot(com_slice, com_row, 'r+', markersize=15, markeredgewidth=3)
        axes[1, 1].set_xlabel('Slice (Z)')
        axes[1, 1].set_ylabel('Row (Y)')
        axes[1, 1].set_title('Tumor Extent (Z-Y view)')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].scatter(tumor_coords[:, 0], tumor_coords[:, 2], 
                          c=tumor_coords[:, 1], cmap='viridis', s=1, alpha=0.6)
        axes[1, 2].plot(com_slice, com_col, 'r+', markersize=15, markeredgewidth=3)
        axes[1, 2].set_xlabel('Slice (Z)')
        axes[1, 2].set_ylabel('Column (X)')
        axes[1, 2].set_title('Tumor Extent (Z-X view)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return com_world, com_voxel
    
    def analyze_tumor_properties(self, roi_name="CTV"):
        """Analyze various tumor properties"""
        tumor_voxels, tumor_coords_voxel, tumor_coords_world, tumor_mask = self.extract_tumor_voxels(roi_name)
        
        # Basic statistics
        volume_mm3 = len(tumor_voxels) * np.prod(self.voxel_spacing)
        volume_cm3 = volume_mm3 / 1000
        
        # Intensity statistics
        mean_intensity = np.mean(tumor_voxels)
        std_intensity = np.std(tumor_voxels)
        min_intensity = np.min(tumor_voxels)
        max_intensity = np.max(tumor_voxels)
        
        # Spatial extent
        min_coords = np.min(tumor_coords_world, axis=0)
        max_coords = np.max(tumor_coords_world, axis=0)
        extent = max_coords - min_coords
        
        # Centers of mass
        com_geometric, com_voxel_geom, _ = self.calculate_center_of_mass(roi_name, 'geometric')
        com_intensity, com_voxel_int, _ = self.calculate_center_of_mass(roi_name, 'intensity')
        
        print(f"\n=== Tumor Analysis for {roi_name} ===")
        print(f"Volume: {volume_mm3:.2f} mm³ ({volume_cm3:.3f} cm³)")
        print(f"Number of voxels: {len(tumor_voxels)}")
        print(f"Intensity - Mean: {mean_intensity:.1f}, Std: {std_intensity:.1f}")
        print(f"Intensity - Range: [{min_intensity:.1f}, {max_intensity:.1f}]")
        print(f"Spatial extent: {extent[0]:.1f} × {extent[1]:.1f} × {extent[2]:.1f} mm")
        print(f"Geometric COM: ({com_geometric[0]:.2f}, {com_geometric[1]:.2f}, {com_geometric[2]:.2f}) mm")
        print(f"Intensity COM: ({com_intensity[0]:.2f}, {com_intensity[1]:.2f}, {com_intensity[2]:.2f}) mm")
        
        return {
            'volume_mm3': volume_mm3,
            'volume_cm3': volume_cm3,
            'n_voxels': len(tumor_voxels),
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'intensity_range': (min_intensity, max_intensity),
            'spatial_extent': extent,
            'geometric_com': com_geometric,
            'intensity_com': com_intensity,
            'tumor_mask': tumor_mask
        }

# Example usage
def demo_tumor_extraction():
    """Demonstrate tumor voxel extraction and analysis"""
    
    # Initialize extractor
    # Physics Informed modalities will be added soon.
    # extractor = TumorVoxelExtractor(
    #     cbct_folder="Physics-informed Code/QML/cbct/folder",
    #     rtstruct_file="Physics-informed Code/QML/structure.dcm"
    # )
    
    # Find available ROIs
    # available_rois = [roi.ROIName for roi in extractor.rtstruct.StructureSetROISequence]
    # print("Available ROIs:", available_rois)
    
    # Extract tumor and calculate center of mass
    # com_world, com_voxel = extractor.visualize_tumor_and_com("CTV", method='geometric')
    
    # Comprehensive analysis
    # properties = extractor.analyze_tumor_properties("CTV")
    
    print("Demo function ready - uncomment lines and provide file paths to run")

if __name__ == "__main__":
    demo_tumor_extraction()