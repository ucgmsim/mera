import pandas as pd

residual_df = pd.read_csv('/home/arr65/src/Andrew_test_code/res_df.csv')
final_mask2D = residual_df.notnull()

# Define the minimum number of records
min_num_records = 4

# Only need to consider one column as all columns in final_mask2D will be the same
final_mask1D = final_mask2D.iloc[:,0]

# Iterate until the DataFrame before and after the masking operation are the same
iteration_counter = 0
# Create a copy of the original DataFrame to compare with the masked DataFrame.
residual_df_prev = residual_df.copy()
while True:
    iteration_counter += 1

    # Use transform on the groupby object to create a mask of the same length as the original DataFrame.
    # Then use iloc to get the first column as all columns are the same so we only need one.
    mask_stat_id = residual_df.groupby('stat_id').transform('count').iloc[:,0] >= min_num_records
    mask_event_id = residual_df.groupby('event_id').transform('count').iloc[:,0] >= min_num_records

    # Combine the station and event masks
    mask = mask_stat_id & mask_event_id

    # Update the final mask. final_mask1D and mask may have different lengths if several iterations are required
    # so we need to use the index of the original DataFrame to update the final mask
    final_mask1D[residual_df.index] = mask

    residual_df = residual_df[mask]

    # If the DataFrame before and after the masking operation are the same, break the loop
    if residual_df.equals(residual_df_prev):
        break

    # Update residual_df_prev for the next iteration
    residual_df_prev = residual_df.copy()

# Copy final_mask1D to all columns in final_mask2D
for col in final_mask2D.columns:
    final_mask2D[col] = final_mask1D

print(f'The number of iterations is: {iteration_counter}')

# Calculate the number of rows removed
num_rows_removed = final_mask2D.shape[0] - residual_df.shape[0]

# Print the number of rows removed
print(f'Removed {num_rows_removed} records (required {iteration_counter} iterations).')

