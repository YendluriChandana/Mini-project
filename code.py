Code 1:
import os, math, json, hashlib, random, csv, shutil
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


OUT_DIR = '/content/advanced_dqn_step_artifacts'
if os.path.exists('/mnt/data'):
    OUT_DIR = '/mnt/data/advanced_dqn_step_artifacts'
os.makedirs(OUT_DIR, exist_ok=True)
print('Artifacts folder:', OUT_DIR)




Code 2:


# Cell 3: Quiz - Expected short answers (printable)
answers = {
  1: "Training curve = episode reward over time; shows learning progress.",
  2: "Offload when uncertain to use a larger edge model for more reliable decision at cost of latency.",
  3: "Ledger provides tamper-evident record of critical decisions for auditability."
}
for q,a in answers.items():
    print(f"{q}. {a}\n")


def render_image(self, scale=40):
    img = Image.new('RGB', (self.size*scale, self.size*scale), 'white')
    draw = ImageDraw.Draw(img)


    for r in range(self.size):
        for c in range(self.size):
            x1,y1 = c*scale, r*scale
            x2,y2 = x1+scale, y1+scale
            draw.rectangle([x1,y1,x2,y2], outline='black')


    for (r,c) in self.obstacles:
        draw.rectangle([c*scale, r*scale, (c+1)*scale, (r+1)*scale], fill='black')


    gr,gc = self.goal
    draw.rectangle([gc*scale, gr*scale, (gc+1)*scale, (gr+1)*scale], fill='green')


    r,c = self.pos
    draw.ellipse([c*scale+8, r*scale+8, (c+1)*scale-8, (r+1)*scale-8], fill='red')


    return img







Code 3:
def draw_grid(env, path, fname, cell=60, show=True):
    size = env.size * cell
    img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(img)


    for r in range(env.size):
        for c in range(env.size):
            x0 = c*cell; y0 = r*cell; x1 = x0+cell; y1 = y0+cell
            draw.rectangle([x0,y0,x1,y1], outline=(180,180,180))


    for (r,c) in env.obstacles:
        draw.rectangle([c*cell+8, r*cell+8, c*cell+cell-8, r*cell+cell-8], fill=(200,50,50))


    for i,(r,c) in enumerate(path):
        cx = c*cell + cell//2; cy = r*cell + cell//2; rad = cell//8
        color = (50,150,50) if i==0 else (20,20,200) if i==len(path)-1 else (30,120,200)
        draw.ellipse([cx-rad, cy-rad, cx+rad, cy+rad], fill=color)


    img.save(fname)


    if show:
        plt.imshow(img)
        plt.axis('off')
        plt.show()


    return fname


Code 4:

import math
import numpy as np


def glorot_init(in_d, out_d, rng):
    limit = math.sqrt(6.0/(in_d+out_d))
    return rng.uniform(-limit, limit, size=(in_d, out_d))
Code 5:
class TinyMLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.RandomState(seed)
        self.sizes = sizes
        self.params = {}
        for i in range(len(sizes)-1):
            self.params[f'W{i}'] = glorot_init(sizes[i], sizes[i+1], rng)
            self.params[f'b{i}'] = np.zeros(sizes[i+1])


    def predict(self, x):
        a = np.array(x)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        L = len(self.sizes) - 1
        for i in range(L-1):
            a = np.tanh(a.dot(self.params[f'W{i}']) + self.params[f'b{i}'])
        z = a.dot(self.params[f'W{L-1}']) + self.params[f'b{L-1}']
        return z




local = TinyMLP([env.size*env.size+2, 16, env.action_space], seed=6)
edge  = TinyMLP([env.size*env.size+2, 64, 32, env.action_space], seed=7)


s = env.reset()
q_local = local.predict(s)[0]
q_edge  = edge.predict(s)[0]


print('q_local:', q_local)
print('q_edge:', q_edge)


plt.figure(figsize=(5,2.5))
plt.bar(range(len(q_local)), q_local, alpha=0.7, label='local')
plt.bar(range(len(q_edge)), q_edge, alpha=0.4, label='edge')
plt.legend()
plt.title('Step 04: Local vs Edge Q-values')
plt.tight_layout()


save_path = os.path.join(OUT_DIR,'step04_q_compare.png')
plt.savefig(save_path)
plt.show()          # 🔥 THIS LINE WAS MISSING
plt.close()


print('Saved Q compare image:', save_path)


Code 5:


from PIL import Image, ImageDraw


def draw_grid(env, path, save_path):
    cell = 60
    W = env.size * cell
    H = env.size * cell


    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)


    # 🔲 DRAW GRID OUTLINE (THIS WAS MISSING)
    for i in range(env.size):
        for j in range(env.size):
            x1 = j * cell
            y1 = i * cell
            x2 = x1 + cell
            y2 = y1 + cell
            d.rectangle([x1, y1, x2, y2], outline='black', width=2)


    # DRAW PATH
    for (r, c) in path:
        cx = c * cell + cell // 2
        cy = r * cell + cell // 2
        d.ellipse(
            [cx-10, cy-10, cx+10, cy+10],
            fill='red'
        )


    img.save(save_path)
    return save_path




import os, math, random, csv, json, hashlib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display






OUT_DIR = '/content/output'
os.makedirs(OUT_DIR, exist_ok=True)
print("Output folder ready:", OUT_DIR)


Code 6:


class GridDroneEnv:
    def __init__(self, size=6, start=(0,0), goal=(5,5)):
        self.size = size
        self.start = start
        self.goal = goal
        self.action_space = 4
        self.reset()


    def reset(self):
        self.pos = self.start
        return self._state()
Code 7:
    def _state(self):
        s = np.zeros(self.size*self.size + 2)
        s[self.pos[0]*self.size + self.pos[1]] = 1
        s[-2:] = self.pos
        return s


    # ⚠️ NOTE: move(), NOT step()
    def move(self, a):
        r, c = self.pos
        if a == 0 and r > 0: r -= 1
        if a == 1 and r < self.size-1: r += 1
        if a == 2 and c > 0: c -= 1
        if a == 3 and c < self.size-1: c += 1
        self.pos = (r, c)
        done = self.pos == self.goal
        reward = 10 if done else -1
        return self._state(), reward, done


    def render_ascii(self, path):
        g = [['.' for _ in range(self.size)] for _ in range(self.size)]
        for r,c in path: g[r][c] = '*'
        g[self.goal[0]][self.goal[1]] = 'G'
        return '\n'.join(' '.join(row) for row in g)


print("Environment class defined")




Code 8:


def glorot_init(in_d, out_d, rng):
    limit = math.sqrt(6.0/(in_d+out_d))
    return rng.uniform(-limit, limit, size=(in_d, out_d))


class TinyMLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.RandomState(seed)
        self.sizes = sizes
        self.params = {}
        for i in range(len(sizes)-1):
            self.params[f'W{i}'] = glorot_init(sizes[i], sizes[i+1], rng)
            self.params[f'b{i}'] = np.zeros(sizes[i+1])


    def predict(self, x):
        a = np.array(x)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        for i in range(len(self.sizes)-2):
            a = np.tanh(a @ self.params[f'W{i}'] + self.params[f'b{i}'])
        return a @ self.params[f'W{len(self.sizes)-2}'] + self.params[f'b{len(self.sizes)-2}']


print("TinyMLP defined")








Code 9:


def draw_grid(env, path, fname):
    cell = 60
    img = Image.new('RGB', (env.size*cell, env.size*cell), 'white')
    d = ImageDraw.Draw(img)


    # grid lines
    for i in range(env.size+1):
        d.line((0, i*cell, env.size*cell, i*cell), fill='black')
        d.line((i*cell, 0, i*cell, env.size*cell), fill='black')


    # path
    for r,c in path:
        d.rectangle(
            [c*cell+10, r*cell+10, c*cell+cell-10, r*cell+cell-10],
            fill=(135,206,235)
        )


    # goal
    gr,gc = env.goal
    d.rectangle(
        [gc*cell+10, gr*cell+10, gc*cell+cell-10, gr*cell+cell-10],
        fill='green'
    )


    img.save(fname)
    return fname




s = env.reset()
path = [env.pos]
done = False


while not done:
    a = int(np.argmax(local.predict(s)[0]))
    s, r, done = env.move(a)
    path.append(env.pos)


pfile = draw_grid(env, path, os.path.join(OUT_DIR, 'step06_path.png'))


display(Image.open(pfile))
print("\nASCII:\n")
print(env.render_ascii(path))


Code 10:




from IPython.display import display
from PIL import Image


ledger_path = os.path.join(OUT_DIR, 'step07_ledger.csv')


with open(ledger_path, 'w') as f:
    f.write('block,ts,tx_count,hash,prev\n')


prev = '0'*64
for i in range(3):
    txs = [{'id':j, 'h': hashlib.sha256(f'{i}-{j}'.encode()).hexdigest()} for j in range(4)]
    body = json.dumps({'i':i, 'txs':txs})
    h = hashlib.sha256((prev + body).encode()).hexdigest()
    with open(ledger_path, 'a') as f:
        f.write(f'{i},{datetime.utcnow().isoformat()},{len(txs)},{h},{prev}\n')
    prev = h


print('Saved ledger CSV:', ledger_path)


# Create preview image
with open(ledger_path,'r') as f:
    lines = f.read().splitlines()[:6]


img = Image.new('RGB', (900, 140), 'white')
d = ImageDraw.Draw(img)
d.text((10,10),'Ledger preview (first lines):', fill='black')


y = 32
for L in lines:
    d.text((10,y), L, fill='black')
    y += 18


img_path = os.path.join(OUT_DIR,'step07_ledger_preview.png')
img.save(img_path)


#  THIS IS WHAT WAS MISSING
display(img)


print('Saved ledger preview image:', img_path)






Code 11:


from IPython.display import display
from PIL import Image


replay_csv = os.path.join(OUT_DIR, 'step08_replay_sample.csv')


with open(replay_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['state_snippet','action','reward','done'])
    for i in range(20):
        w.writerow([str(list(range(5)))+'...', random.randrange(env.action_space),
                    random.uniform(-5,20), False])


print('Saved replay sample CSV:', replay_csv)


# Preview image
with open(replay_csv,'r') as f:
    txt = f.read().splitlines()[:8]


img = Image.new('RGB', (900,160), 'white')
d = ImageDraw.Draw(img)
d.text((10,10),'Replay sample (first lines):', fill='black')


y = 30
for L in txt:
    d.text((10,y), L, fill='black')
    y += 16


img_path = os.path.join(OUT_DIR,'step08_replay_preview.png')
img.save(img_path)


#  THIS WAS MISSING
display(img)


print('Saved replay preview image:', img_path)




Code 12:


from IPython.display import display
from PIL import Image


env2 = GridDroneEnv(size=7, obstacles={(2,2),(3,2),(4,4),(1,5)}, start=(0,0), goal=(6,6))
obs_dim = env2.size*env2.size + 2
net_final = TinyMLP([obs_dim, 128, 64, env2.action_space], seed=21)


rewards_final = []


for ep in range(1,61):
    s = env2.reset(); done=False; ep_r = 0
    while not done:
        a = int(np.argmax(net_final.predict(s)[0]))
        if random.random() < 0.12:
            a = random.randrange(env2.action_space)
        s, r, done, _ = env2.step(a)
        ep_r += r
    rewards_final.append(ep_r)
    if ep % 10 == 0:
        print('Ep', ep, 'mean reward so far', sum(rewards_final)/len(rewards_final))


# 🔹 Reward curve
plt.figure(figsize=(6,3))
plt.plot(rewards_final)
plt.title('Final-run: Reward curve')
curve_path = os.path.join(OUT_DIR,'step11_final_curve.png')
plt.savefig(curve_path)
plt.show()              # MISSING LINE
plt.close()


# 🔹 Final path
s = env2.reset()
path = [env2.pos]
d = False
while not d:
    a = int(np.argmax(net_final.predict(s)[0]))
    s, _, d, _ = env2.step(a)
    path.append(env2.pos)


path_img = draw_grid(env2, path, os.path.join(OUT_DIR,'step11_final_path.png'))
display(Image.open(path_img))   #  MISSING LINE


# 🔹 Stats
with open(os.path.join(OUT_DIR,'step11_final_stats.txt'),'w') as f:
    f.write('episodes:60\nmean_reward:'+str(sum(rewards_final)/len(rewards_final)))


print('Saved final artifacts in', OUT_DIR)




Code 13:

from IPython.display import display
from PIL import Image, ImageFont


def save_comparison_venn(fname):
    size = 640
    img = Image.new('RGB', (size,size), 'white')
    d = ImageDraw.Draw(img)
    cx1, cy = 200, 320; cx2 = 420; r = 150
    d.ellipse([cx1-r, cy-r, cx1+r, cy+r], outline=(70,130,180), width=6)
    d.ellipse([cx2-r, cy-r, cx2+r, cy+r], outline=(180,80,120), width=6)
    try:
        font = ImageFont.truetype('DejaVuSans-Bold.ttf', 16)
    except:
        font = ImageFont.load_default()
    d.text((cx1-60, cy-170), 'Previous work', fill=(70,130,180), font=font)
    d.text((cx2-40, cy-170), 'Our work', fill=(180,80,120), font=font)
    left=['DRL navigation','Simulated envs','Stability tricks']
    right=['Edge offload','Blockchain logging','End-to-end code']
    overlap=['Navigation + offload','Auditability']
    y = cy-30
    for t in left:
        d.text((50,y), '• '+t, fill='black', font=font); y+=20
    y = cy-30
    for t in right:
        d.text((430,y), '• '+t, fill='black', font=font); y+=20
    y = cy+70
    for t in overlap:
        d.text((265,y), '• '+t, fill='black', font=font); y+=20
    img.save(fname)
    return fname


Code 14:


def save_comparison_table(fname):
    cols = ['Feature','Previous works','This project']
    rows = [
        ('Navigation by DRL','Yes','Yes'),
        ('Uncertainty-aware offload','Sometimes','Yes'),
        ('Blockchain audit','Some proposals','Yes (simulated)'),
        ('Reproducible code','Rare','Yes (Colab)'),
        ('Stepwise images for paper','No','Yes'),
    ]
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 14)
    except:
        font = ImageFont.load_default()
    padding = 12; col_w = [240, 220, 220]; row_h = 36
    w = sum(col_w) + 2*padding
    h = (len(rows)+1)*row_h + 2*padding
    img = Image.new('RGB', (w, h), 'white')
    d = ImageDraw.Draw(img)


    x = padding; y = padding
    for i,c in enumerate(cols):
        d.rectangle([x, y, x+col_w[i], y+row_h],
                    fill=(200,200,255) if i==0 else (230,230,230),
                    outline='black')
        d.text((x+8, y+8), c, fill='black', font=font)
        x += col_w[i]
    y += row_h


    for r in rows:
        x = padding
        for i,cell in enumerate(r):
            d.rectangle([x, y, x+col_w[i], y+row_h], outline='black')
            d.text((x+8, y+8), str(cell), fill='black', font=font)
            x += col_w[i]
        y += row_h


    img.save(fname)
    return fname


vfname = save_comparison_venn(os.path.join(OUT_DIR,'comparison_venn.png'))
tfname = save_comparison_table(os.path.join(OUT_DIR,'comparison_table.png'))


#  DISPLAY (THIS WAS MISSING)
display(Image.open(vfname))
display(Image.open(tfname))


print('Saved comparison images:', vfname, tfname)
