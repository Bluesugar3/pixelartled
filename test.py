"""Minimal Snake (Windows console friendly).
Controls: Arrow keys or WASD. Quit: Q.
Single file, stdlib only.
"""
import os, time, random, msvcrt

H,W=15,30
snake=[(H//2, W//2+i) for i in range(3)][::-1]
dir=(0,1)
food=(random.randrange(H), random.randrange(W))
score=0

def place_food():
	global food
	while True:
		f=(random.randrange(H),random.randrange(W))
		if f not in snake:
			food=f; return

def draw():
	os.system('cls')
	print(f'Score: {score}')
	rows=[]
	body=set(snake[1:])
	for y in range(H):
		r=[]
		for x in range(W):
			if (y,x)==snake[0]: r.append('O')
			elif (y,x)==food: r.append('*')
			elif (y,x) in body: r.append('#')
			else: r.append(' ')
		rows.append('|' + ''.join(r) + '|')
	print('+'+'-'*W+'+')
	print('\n'.join(rows))
	print('+'+'-'*W+'+')
	print('Arrow/WASD, Q quit')

def read_dir():
	global dir
	if not msvcrt.kbhit(): return
	c=msvcrt.getch()
	if c in b'wasdWASD':
		mapping={b'w':(-1,0),b'a':(0,-1),b's':(1,0),b'd':(0,1)}
		nd=mapping[c.lower()]
	elif c==b'\xe0':  # arrow prefix
		k=msvcrt.getch()
		arrows={b'H':(-1,0),b'K':(0,-1),b'P':(1,0),b'M':(0,1)}
		if k in arrows: nd=arrows[k]
		else: return
	elif c in (b'q',b'Q'):
		raise SystemExit
	else: return
	if nd!=(-dir[0],-dir[1]): dir=nd

draw(); last=time.time()
while True:
	read_dir()
	if time.time()-last<0.12: continue
	last=time.time()
	ny=(snake[0][0]+dir[0])%H; nx=(snake[0][1]+dir[1])%W
	if (ny,nx) in snake:
		break
	snake.insert(0,(ny,nx))
	if (ny,nx)==food:
		score+=1; place_food()
	else:
		snake.pop()
	draw()

print('GAME OVER. Final score:',score)





