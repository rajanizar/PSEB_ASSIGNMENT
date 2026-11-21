# Dataset Schema

- **Rows:** 5,150
- **Columns:** 12
- **Raw Integrity:** Files in `data/raw/` remain untouched.

| Column          | Dtype   |   Missing Count | Missing %   |   Unique Count |
|:----------------|:--------|----------------:|:------------|---------------:|
| Transaction_ID  | str     |               0 | 0.00%       |           5000 |
| Sale_Date       | str     |               0 | 0.00%       |           2770 |
| Make            | str     |               0 | 0.00%       |             46 |
| Model           | str     |               0 | 0.00%       |            225 |
| Vehicle_Type    | str     |               0 | 0.00%       |             27 |
| Fuel_Type       | str     |             419 | 8.14%       |             34 |
| Year            | int64   |               0 | 0.00%       |             12 |
| Price_USD       | str     |             444 | 8.62%       |           4573 |
| Units_Sold      | float64 |             416 | 8.08%       |             26 |
| Total_Sales_USD | float64 |             409 | 7.94%       |           4593 |
| Region          | str     |             516 | 10.02%      |             32 |
| Sales_Rep       | str     |             479 | 9.30%       |             56 |
