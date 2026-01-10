import os
import sys
import argparse
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import griddata, NearestNDInterpolator
from scipy.ndimage import zoom
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Optional PyTorch import for .pt file saving
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class INSATRegridder:
    """Class to handle INSAT-3D data regridding to 720x720 resolution."""
    
    def __init__(self, target_size=720, interpolation_method='linear', amv_max_distance=50000, save_pt=False):
        """
        Initialize the regridder.
        
        Args:
            target_size (int): Target grid size (720 for 720x720)
            interpolation_method (str): Interpolation method ('linear', 'nearest', 'cubic')
            amv_max_distance (float): Maximum distance for AMV interpolation
            save_pt (bool): Whether to save data as PyTorch .pt files
        """
        self.target_size = target_size
        self.interpolation_method = interpolation_method
        self.amv_max_distance = amv_max_distance
        self.save_pt = save_pt
        
        if self.save_pt and not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for .pt file saving. Install with: pip install torch")
        
    def create_target_grid(self, x_coords, y_coords):
        """
        Create uniform target grid coordinates.
        
        Args:
            x_coords (array): Original x coordinates
            y_coords (array): Original y coordinates
            
        Returns:
            tuple: Target x and y coordinate arrays
        """
        # Create uniform grid covering the same spatial extent
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        target_x = np.linspace(x_min, x_max, self.target_size)
        target_y = np.linspace(y_min, y_max, self.target_size)
        
        return np.meshgrid(target_x, target_y)
    
    def regrid_dense_variable(self, data, source_shape, target_shape):
        """
        Regrid dense variables using zoom interpolation.
        
        Args:
            data (array): Input data array
            source_shape (tuple): Original data shape
            target_shape (tuple): Target data shape
            
        Returns:
            array: Regridded data
        """
        if np.all(np.isnan(data)):
            return np.full(target_shape, np.nan)
            
        # Calculate zoom factors
        zoom_factors = (target_shape[0] / source_shape[0], 
                       target_shape[1] / source_shape[1])
        
        # Handle NaN values by creating a mask
        valid_mask = ~np.isnan(data)
        
        if np.sum(valid_mask) == 0:
            return np.full(target_shape, np.nan)
        
        # Zoom the valid data
        try:
            # For data with many NaNs, use nearest neighbor to avoid interpolation artifacts
            if np.sum(valid_mask) / data.size < 0.5:
                regridded = zoom(data, zoom_factors, order=0, mode='nearest', prefilter=False)
            else:
                regridded = zoom(data, zoom_factors, order=1, mode='nearest', prefilter=False)
            
            return regridded
            
        except Exception as e:
            print(f"Warning: Failed to regrid variable, using nearest neighbor: {e}")
            return zoom(data, zoom_factors, order=0, mode='nearest', prefilter=False)
    
    def regrid_sparse_amv_vectors(self, u_comp, v_comp, x_coords, y_coords, target_x_grid, target_y_grid, max_distance_threshold=50000):
        """
        Regrid sparse AMV vector data using interpolation with distance threshold.
        
        Args:
            u_comp (array): U component of wind vectors
            v_comp (array): V component of wind vectors
            x_coords (array): Original x coordinates
            y_coords (array): Original y coordinates
            target_x_grid (array): Target x coordinate grid
            target_y_grid (array): Target y coordinate grid
            max_distance_threshold (float): Maximum distance (in coordinate units) for interpolation.
                                          Points beyond this distance from valid data will be NaN.
            
        Returns:
            tuple: Regridded u and v components
        """
        # Find valid (non-NaN) vector data points
        valid_mask = ~(np.isnan(u_comp) | np.isnan(v_comp))
        
        if np.sum(valid_mask) == 0:
            # No valid data, return NaN arrays and mask indicating all invalid
            return (np.full((self.target_size, self.target_size), np.nan),
                   np.full((self.target_size, self.target_size), np.nan),
                   np.ones((self.target_size, self.target_size), dtype=bool))  # All pixels are invalid
        
        # Get coordinates of valid points
        y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        valid_x = x_grid[valid_mask]
        valid_y = y_grid[valid_mask]
        valid_u = u_comp[valid_mask]
        valid_v = v_comp[valid_mask]
        
        # Create coordinate arrays for interpolation
        points = np.column_stack((valid_x.flatten(), valid_y.flatten()))
        target_points = np.column_stack((target_x_grid.flatten(), target_y_grid.flatten()))
        
        try:
            # Interpolate u component
            if self.interpolation_method == 'nearest':
                interpolator = NearestNDInterpolator(points, valid_u)
                u_regridded = interpolator(target_points).reshape(self.target_size, self.target_size)
                interpolator = NearestNDInterpolator(points, valid_v)
                v_regridded = interpolator(target_points).reshape(self.target_size, self.target_size)
                
                # Apply distance threshold for nearest neighbor interpolation
                from scipy.spatial import cKDTree
                
                # Build KDTree for efficient nearest neighbor search
                tree = cKDTree(points)
                
                # Find distances to nearest neighbors for all target points
                distances, _ = tree.query(target_points)
                distance_mask = distances > max_distance_threshold
                
                # Set values beyond threshold to NaN
                u_regridded_flat = u_regridded.flatten()
                v_regridded_flat = v_regridded.flatten()
                u_regridded_flat[distance_mask] = np.nan
                v_regridded_flat[distance_mask] = np.nan
                
                u_regridded = u_regridded_flat.reshape(self.target_size, self.target_size)
                v_regridded = v_regridded_flat.reshape(self.target_size, self.target_size)
                
                # Return the regridded data and the distance mask for consistent mask generation
                return u_regridded, v_regridded, distance_mask.reshape(self.target_size, self.target_size)
                
            else:
                u_regridded = griddata(points, valid_u, target_points, 
                                     method=self.interpolation_method, 
                                     fill_value=np.nan)
                u_regridded = u_regridded.reshape(self.target_size, self.target_size)
                
                # Interpolate v component
                v_regridded = griddata(points, valid_v, target_points, 
                                     method=self.interpolation_method, 
                                     fill_value=np.nan)
                v_regridded = v_regridded.reshape(self.target_size, self.target_size)
                
                # For non-nearest methods, distance mask is not applied, return None
                return u_regridded, v_regridded, None
            
            return u_regridded, v_regridded, distance_mask.reshape(self.target_size, self.target_size) if 'distance_mask' in locals() else None
            
        except Exception as e:
            print(f"Warning: AMV interpolation failed, using nearest neighbor: {e}")
            # Fallback to nearest neighbor with distance threshold
            from scipy.spatial import cKDTree
            
            interpolator = NearestNDInterpolator(points, valid_u)
            u_regridded = interpolator(target_points).reshape(self.target_size, self.target_size)
            interpolator = NearestNDInterpolator(points, valid_v)
            v_regridded = interpolator(target_points).reshape(self.target_size, self.target_size)
            
            # Apply distance threshold
            tree = cKDTree(points)
            distances, _ = tree.query(target_points)
            distance_mask = distances > max_distance_threshold
            
            u_regridded_flat = u_regridded.flatten()
            v_regridded_flat = v_regridded.flatten()
            u_regridded_flat[distance_mask] = np.nan
            v_regridded_flat[distance_mask] = np.nan
            
            u_regridded = u_regridded_flat.reshape(self.target_size, self.target_size)
            v_regridded = v_regridded_flat.reshape(self.target_size, self.target_size)
            
            return u_regridded, v_regridded, distance_mask.reshape(self.target_size, self.target_size)
    
    def save_as_pytorch_tensor(self, nc_file_path, pt_file_path):
        """
        Convert NetCDF file to PyTorch tensor format and save as .pt file.
        
        Args:
            nc_file_path (str): Path to the NetCDF file
            pt_file_path (str): Path to save the .pt file
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for .pt file saving")
        
        print(f"  Converting to PyTorch tensor: {os.path.basename(pt_file_path)}")
        
        with Dataset(nc_file_path, 'r') as ds:
            # Define the expected order of variables (interleaved data + mask pairs = 16 layers)
            data_vars = ['AMV_IRW_ucomp', 'AMV_IRW_vcomp', 'L2B_SST', 'L2B_OLR', 
                        'L2B_UTH', 'L2B_CTT', 'TPW_SA1_totH2O', 'TPW_SB1_totH2O']
            
            # Create tensor with shape (16, height, width) for 16 channels
            tensor_data = np.zeros((16, self.target_size, self.target_size), dtype=np.float32)
            
            # Fill data and mask layers in interleaved order (data, mask, data, mask, ...)
            channel_idx = 0
            for var_name in data_vars:
                # Fill data layer
                if var_name in ds.variables:
                    data = ds.variables[var_name][:]
                    # Replace NaN with 0 for tensor (use masks to identify valid data)
                    data = np.nan_to_num(data, nan=0.0)
                    tensor_data[channel_idx] = data.astype(np.float32)
                else:
                    print(f"    Warning: {var_name} not found, filling with zeros")
                
                # Fill corresponding mask layer
                mask_name = f"{var_name}_mask"
                if mask_name in ds.variables:
                    mask_data = ds.variables[mask_name][:]
                    tensor_data[channel_idx + 1] = mask_data.astype(np.float32)
                else:
                    print(f"    Warning: {mask_name} not found, filling with zeros")
                
                channel_idx += 2  # Move to next data/mask pair
            
            # Convert to PyTorch tensor
            tensor = torch.from_numpy(tensor_data)
            
            # Create metadata dictionary
            metadata = {
                'shape': tensor.shape,
                'channels': {
                    # Interleaved data/mask pairs
                    0: 'AMV_IRW_ucomp', 1: 'AMV_IRW_ucomp_mask',
                    2: 'AMV_IRW_vcomp', 3: 'AMV_IRW_vcomp_mask', 
                    4: 'L2B_SST', 5: 'L2B_SST_mask',
                    6: 'L2B_OLR', 7: 'L2B_OLR_mask',
                    8: 'L2B_UTH', 9: 'L2B_UTH_mask',
                    10: 'L2B_CTT', 11: 'L2B_CTT_mask',
                    12: 'TPW_SA1_totH2O', 13: 'TPW_SA1_totH2O_mask',
                    14: 'TPW_SB1_totH2O', 15: 'TPW_SB1_totH2O_mask'
                },
                'description': 'INSAT-3D regridded data with interleaved data/mask channel pairs',
                'channel_structure': 'Interleaved: data, mask, data, mask, ... (even=data, odd=mask)',
                'data_channels_info': 'Even channels (0,2,4,...): data values (NaN replaced with 0)',
                'mask_channels_info': 'Odd channels (1,3,5,...): masks (1=valid, 0=invalid)',
                'grid_size': f'{self.target_size}x{self.target_size}',
                'interpolation_method': self.interpolation_method,
                'amv_max_distance': self.amv_max_distance,
                'coordinate_info': {
                    'x_min': float(ds.variables['x'][:].min()),
                    'x_max': float(ds.variables['x'][:].max()),
                    'y_min': float(ds.variables['y'][:].min()),
                    'y_max': float(ds.variables['y'][:].max()),
                }
            }
            
            # Save tensor and metadata
            torch.save({
                'data': tensor,
                'metadata': metadata
            }, pt_file_path)
            
            print(f"    Saved tensor shape: {tensor.shape} to {os.path.basename(pt_file_path)}")
    
    def process_file(self, input_file, output_file):
        """
        Process a single INSAT file and save regridded output.
        
        Args:
            input_file (str): Path to input NetCDF file
            output_file (str): Path to output NetCDF file
        """
        print(f"Processing: {os.path.basename(input_file)}")
        
        # Determine output paths
        pt_file_path = output_file.replace('.nc', '.pt') if self.save_pt else None
        
        # If only saving PT file, create a temporary NC file for conversion
        temp_nc_file = None
        if self.save_pt:
            temp_nc_file = output_file.replace('.nc', '_temp.nc')
            nc_output_path = temp_nc_file
        else:
            nc_output_path = output_file
        
        # Open input file
        with Dataset(input_file, 'r') as ds_in:
            # Read coordinates
            x_coords = ds_in.variables['x'][:]
            y_coords = ds_in.variables['y'][:]
            
            # Create target grid
            target_x_grid, target_y_grid = self.create_target_grid(x_coords, y_coords)
            
            # Create output file
            with Dataset(nc_output_path, 'w', format='NETCDF4') as ds_out:
                # Create dimensions
                ds_out.createDimension('y', self.target_size)
                ds_out.createDimension('x', self.target_size)
                
                # Create coordinate variables
                y_var = ds_out.createVariable('y', 'f8', ('y',))
                x_var = ds_out.createVariable('x', 'f8', ('x',))
                
                # Set coordinate values
                x_var[:] = target_x_grid[0, :]
                y_var[:] = target_y_grid[:, 0]
                
                # Copy coordinate attributes (except _FillValue since coordinates shouldn't have fill values)
                for attr_name in ds_in.variables['x'].ncattrs():
                    if attr_name != '_FillValue':
                        x_var.setncattr(attr_name, ds_in.variables['x'].getncattr(attr_name))
                for attr_name in ds_in.variables['y'].ncattrs():
                    if attr_name != '_FillValue':
                        y_var.setncattr(attr_name, ds_in.variables['y'].getncattr(attr_name))
                
                # Process each variable
                amv_variables = ['AMV_IRW_ucomp', 'AMV_IRW_vcomp']
                other_variables = [var for var in ds_in.variables.keys() 
                                 if var not in ['x', 'y', 'crs'] + amv_variables]
                
                # Process AMV vectors together
                if 'AMV_IRW_ucomp' in ds_in.variables and 'AMV_IRW_vcomp' in ds_in.variables:
                    print("  Processing AMV vectors...")
                    u_comp = ds_in.variables['AMV_IRW_ucomp'][:]
                    v_comp = ds_in.variables['AMV_IRW_vcomp'][:]
                    
                    u_regridded, v_regridded, amv_distance_mask = self.regrid_sparse_amv_vectors(
                        u_comp, v_comp, x_coords, y_coords, target_x_grid, target_y_grid, self.amv_max_distance
                    )
                    
                    # Create AMV output variables
                    u_var = ds_out.createVariable('AMV_IRW_ucomp', 'f4', ('y', 'x'), 
                                                  fill_value=np.nan, zlib=True, complevel=6)
                    v_var = ds_out.createVariable('AMV_IRW_vcomp', 'f4', ('y', 'x'), 
                                                  fill_value=np.nan, zlib=True, complevel=6)
                    
                    u_var[:] = u_regridded
                    v_var[:] = v_regridded
                    
                    # Create mask variables for AMV vectors
                    u_mask_var = ds_out.createVariable('AMV_IRW_ucomp_mask', 'i1', ('y', 'x'), 
                                                       zlib=True, complevel=6)
                    v_mask_var = ds_out.createVariable('AMV_IRW_vcomp_mask', 'i1', ('y', 'x'), 
                                                       zlib=True, complevel=6)
                    
                    # Set mask values (1 for valid data, 0 for NaN or beyond distance threshold)
                    if amv_distance_mask is not None:
                        # For nearest neighbor with distance threshold
                        u_mask_data = (~np.isnan(u_regridded) & ~amv_distance_mask).astype(np.int8)
                        v_mask_data = (~np.isnan(v_regridded) & ~amv_distance_mask).astype(np.int8)
                    else:
                        # For other interpolation methods
                        u_mask_data = (~np.isnan(u_regridded)).astype(np.int8)
                        v_mask_data = (~np.isnan(v_regridded)).astype(np.int8)
                    
                    u_mask_var[:] = u_mask_data
                    v_mask_var[:] = v_mask_data
                    
                    # Add mask attributes
                    u_mask_var.setncattr('long_name', 'Mask for AMV_IRW_ucomp (1=valid, 0=missing/beyond_threshold)')
                    u_mask_var.setncattr('description', f'Binary mask indicating valid data pixels within {self.amv_max_distance} units of source data')
                    v_mask_var.setncattr('long_name', 'Mask for AMV_IRW_vcomp (1=valid, 0=missing/beyond_threshold)')
                    v_mask_var.setncattr('description', f'Binary mask indicating valid data pixels within {self.amv_max_distance} units of source data')
                    
                    # Copy attributes (except _FillValue since it's already set)
                    for attr_name in ds_in.variables['AMV_IRW_ucomp'].ncattrs():
                        if attr_name != '_FillValue':
                            u_var.setncattr(attr_name, ds_in.variables['AMV_IRW_ucomp'].getncattr(attr_name))
                    for attr_name in ds_in.variables['AMV_IRW_vcomp'].ncattrs():
                        if attr_name != '_FillValue':
                            v_var.setncattr(attr_name, ds_in.variables['AMV_IRW_vcomp'].getncattr(attr_name))
                
                # Process other variables
                for var_name in other_variables:
                    if var_name in ds_in.variables:
                        print(f"  Processing {var_name}...")
                        original_data = ds_in.variables[var_name][:]
                        
                        # Regrid the data
                        regridded_data = self.regrid_dense_variable(
                            original_data, 
                            original_data.shape, 
                            (self.target_size, self.target_size)
                        )
                        
                        # Create output variable with proper fill value
                        out_var = ds_out.createVariable(var_name, 'f4', ('y', 'x'), 
                                                       fill_value=np.nan, zlib=True, complevel=6)
                        out_var[:] = regridded_data
                        
                        # Create corresponding mask variable
                        mask_var_name = f"{var_name}_mask"
                        mask_var = ds_out.createVariable(mask_var_name, 'i1', ('y', 'x'), 
                                                        zlib=True, complevel=6)
                        
                        # Set mask values (1 for valid data, 0 for NaN)
                        mask_var[:] = (~np.isnan(regridded_data)).astype(np.int8)
                        
                        # Add mask attributes
                        mask_var.setncattr('long_name', f'Mask for {var_name} (1=valid, 0=missing)')
                        mask_var.setncattr('description', 'Binary mask indicating valid data pixels')
                        mask_var.setncattr('units', 'dimensionless')
                        
                        # Copy attributes (except _FillValue since it's already set)
                        for attr_name in ds_in.variables[var_name].ncattrs():
                            if attr_name != '_FillValue':
                                out_var.setncattr(attr_name, ds_in.variables[var_name].getncattr(attr_name))
                
                # Copy global attributes
                for attr_name in ds_in.ncattrs():
                    ds_out.setncattr(attr_name, ds_in.getncattr(attr_name))
                
                # Add processing information
                ds_out.setncattr('regridding_info', f'Regridded to {self.target_size}x{self.target_size} using {self.interpolation_method} interpolation')
                ds_out.setncattr('amv_max_distance', f'AMV interpolation limited to {self.amv_max_distance} coordinate units')
                ds_out.setncattr('regridding_timestamp', str(np.datetime64('now')))
        
        # Handle output file creation based on save_pt flag
        if self.save_pt:
            # Convert to PyTorch tensor and clean up temp file
            self.save_as_pytorch_tensor(nc_output_path, pt_file_path)
            # Remove temporary NetCDF file
            if temp_nc_file and os.path.exists(temp_nc_file):
                os.remove(temp_nc_file)
                print(f"  Saved: {os.path.basename(pt_file_path)} (PT format only)")
        else:
            print(f"  Saved: {os.path.basename(output_file)} (NetCDF format)")


def main():
    """Main function to handle command line arguments and process files."""
    parser = argparse.ArgumentParser(description='Regrid INSAT-3D data to 720x720 resolution')
    parser.add_argument('--input_dir', type=str, default='~/Downloads/dem',
                       help='Directory containing INSAT NetCDF files')
    parser.add_argument('--output_dir', type=str, default='./regridded_insat',
                       help='Output directory for regridded files')
    parser.add_argument('--target_size', type=int, default=720,
                       help='Target grid size (default: 720)')
    parser.add_argument('--interpolation', type=str, default='linear',
                       choices=['linear', 'nearest', 'cubic'],
                       help='Interpolation method for AMV vectors')
    parser.add_argument('--amv_max_distance', type=float, default=50000,
                       help='Maximum distance (in coordinate units) for AMV interpolation. Beyond this distance, pixels will be NaN.')
    parser.add_argument('--save_pt', action='store_true',
                       help='Save data as PyTorch .pt files in addition to NetCDF (requires torch)')
    parser.add_argument('--pattern', type=str, default='INSAT3DR_*.nc',
                       help='File pattern to match')
    
    args = parser.parse_args()
    
    # Expand user directory
    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find input files
    input_pattern = os.path.join(input_dir, args.pattern)
    input_files = glob.glob(input_pattern)
    
    if not input_files:
        print(f"No files found matching pattern: {input_pattern}")
        return
    
    print(f"Found {len(input_files)} files to process")
    
    # Initialize regridder
    regridder = INSATRegridder(target_size=args.target_size, 
                              interpolation_method=args.interpolation,
                              amv_max_distance=args.amv_max_distance,
                              save_pt=args.save_pt)
    
    # Process each file
    for input_file in tqdm(input_files, desc="Processing files"):
        try:
            # Generate output filename
            base_name = os.path.basename(input_file)
            output_name = base_name.replace('.nc', f'_regridded_{args.target_size}x{args.target_size}.nc')
            output_file = os.path.join(output_dir, output_name)
            
            # Determine what file to check for existence
            if args.save_pt:
                # When saving PT, check for PT file existence
                pt_file = output_file.replace('.nc', '.pt')
                target_file = pt_file
                target_name = os.path.basename(pt_file)
            else:
                # When saving NC, check for NC file existence
                target_file = output_file
                target_name = os.path.basename(output_file)
            
            # Skip if target output already exists
            if os.path.exists(target_file):
                print(f"  Skipping {base_name} (output {target_name} exists)")
                continue
            
            # Process the file
            regridder.process_file(input_file, output_file)
            
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue
    
    output_format = "PyTorch .pt files" if args.save_pt else "NetCDF .nc files"
    print(f"\nProcessing complete! {output_format} saved to: {output_dir}")


if __name__ == "__main__":
    main()