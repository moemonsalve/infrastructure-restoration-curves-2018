#!/usr/bin/python

FSIZE = (5.5,3.2)

import sys

T = []
D = []

f = open(sys.argv[1])
T = f.readline().strip().split(',')
D = [[] for t in T]
for line in f:
	row = line.strip().split(',')
	if len(row)!=len(T):
		continue
	for i in range(len(row)):
		D[i].append( float(row[i]) )
f.close()

# Let us reorder the data
INDEX = [y for (x,y) in sorted( zip(T,range(len(T))) ) ]
Tnew = [ T[i] for i in INDEX ]
Dnew = [ D[i] for i in INDEX ]
T,D = Tnew,Dnew

#print T
#print D

def estilo(field):
	x = field.upper()
	if   x == 'POWER':	 return 'y-'
	elif x == 'WATER':	 return 'b:'
	elif x == 'GAS':	 return 'm--'
	elif x == 'TELECOM': return 'r:.'
	elif x == 'HOSPITAL':	 return 'g:*'
	elif x == 'FIXED':	 return ':xc'
	elif x == 'MOBILE':	 return ':+r'
	else: return ':k'

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
plt.figure( figsize=FSIZE )
cols = 'brgcykm'
styl = '- : -. --'.split(' ')
mrkr = ['.','^','*']
for i in range(len(T)):
	#plt.plot( D[i], cols[i%7] + mrkr[i%3] + styl[i%4], label=T[i])
	plt.plot( D[i], estilo(T[i]), label=T[i])
plt.xlabel( 'Days, $t$' )
plt.ylabel( 'Functionality, $x_i(t)$' )
plt.legend( loc=4 , fontsize = 13)
plt.tight_layout()
#plt.show()
plt.savefig( 'out.pdf' , bbox_inches='tight')
