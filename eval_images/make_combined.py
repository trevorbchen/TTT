import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

tw = np.array(Image.open('eval_images/recon_tweedie.png'))
dr = np.array(Image.open('eval_images/recon_direct.png'))

print(f'Tweedie image: {tw.shape}')
print(f'Direct image: {dr.shape}')

# Tweedie has 4 cols, direct has 2 cols
tw_h, tw_w = tw.shape[:2]
dr_h, dr_w = dr.shape[:2]
tw_col_w = tw_w // 4
dr_col_w = dr_w // 2

# Extract columns from tweedie: GT(0), CBG(1), DPS(2), Plain(3)
gt_col = tw[:, :tw_col_w]
cbg_tw_col = tw[:, tw_col_w:2*tw_col_w]
dps_col = tw[:, 2*tw_col_w:3*tw_col_w]
plain_col = tw[:, 3*tw_col_w:]

# Extract CBG column from direct (col 1)
cbg_dr_col = dr[:, dr_col_w:]

# Resize direct column to match tweedie dimensions if needed
if cbg_dr_col.shape[0] != tw_h or cbg_dr_col.shape[1] != tw_col_w:
    cbg_dr_pil = Image.fromarray(cbg_dr_col)
    cbg_dr_pil = cbg_dr_pil.resize((tw_col_w, tw_h), Image.LANCZOS)
    cbg_dr_col = np.array(cbg_dr_pil)

# Combine: GT | CBG-tweedie | CBG-direct | DPS | Plain
combined = np.concatenate([gt_col, cbg_tw_col, cbg_dr_col, dps_col, plain_col], axis=1)

# Add column headers
fig, ax = plt.subplots(figsize=(20, 28))
ax.imshow(combined)
ax.axis('off')

labels = ['Ground Truth', 'CBG-tweedie', 'CBG-direct', 'DPS', 'Plain']
for i, label in enumerate(labels):
    x = (i + 0.5) * tw_col_w
    ax.text(x, -10, label, ha='center', va='bottom', fontsize=16, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))

plt.tight_layout()
plt.savefig('eval_images/combined_all_methods.png', dpi=150, bbox_inches='tight',
            facecolor='black', pad_inches=0.3)
plt.close()
print('Saved combined_all_methods.png')
