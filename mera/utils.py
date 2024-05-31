import pandas as pd
import numpy as np

df = pd.read_csv('/home/arr65/src/Andrew_test_code/res_df.csv')

df_original = df.copy()

# Define the minimum number of records
min_num_records = 4

# Create a copy of the original DataFrame to compare with the masked DataFrame
df_prev = df.copy()

# Initialize the counter
iteration_counter = 0

# Initialize the final mask with all True values (same size as the original DataFrame)
final_mask = pd.Series(np.ones(df.shape[0], dtype=bool))

while True:
    # Increment the counter
    iteration_counter += 1

    # Create masks for 'stat_id' and 'event_id'
    mask_stat_id = df.groupby('stat_id')['stat_id'].transform('count') >= min_num_records
    mask_event_id = df.groupby('event_id')['event_id'].transform('count') >= min_num_records

    # Combine the masks
    mask = mask_stat_id & mask_event_id

    # Update the final mask
    final_mask[df.index] = mask

    # Apply the mask to the DataFrame
    df = df[mask]

    # If the DataFrame before and after the masking operation are the same, break the loop
    if df.equals(df_prev):
        break

    # Update df_prev for the next iteration
    df_prev = df.copy()

print(f'The number of iterations is: {iteration_counter}')

# Calculate the number of rows removed
num_rows_removed = df_original.shape[0] - df.shape[0]

# Print the number of rows removed
print(f'The number of rows removed is: {num_rows_removed}')

