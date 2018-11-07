import re, sys
import scipy as sc
import scipy.linalg as sl
import scipy.stats as st

args = sys.argv[1:]
parSilent = '--silent' in args
parLaTeX  = '--latex' in args
parTest   = '--test' in args
parMField = '--meanfield' in args
parStoch  = '--stochastic' in args
fname = [x for x in args if x not in ['--silent','--latex','--test','--meanfield','--stochastic']]

FSIZE = (5.5,3.2)

if len(fname)==0:
	print 'Usage:\n\n\t inter-est.py [options] file1 file2 ...\n'
	print 'Options:\n\t--silent     Do not print convergence data.'
	print '\t--latex      Print the formulae in LaTeX'
	print '\t--test       Print test data (R^2) instead of formulae'
	print '\t--meanfield  Compares the supplied data against the mean'
	print '\t             field reconstruction. Saves to out.png.'
	print '\t--stochastic Compares the supplied data against randomly'
	print '\t             generated curves. Saves to out.png.'
	exit(0)

f = open(fname[0])
T = f.readline().strip().split(',')

S = [[] for title in T]
for line in f:
	row = line.strip().split(',')
	if len(row)<2: continue
	for i in range(len(S)):
		S[i].append(float(row[i]))
f.close()

if not parSilent: print [len(s) for s in S]

# Let us reorder the data
INDEX = [y for (x,y) in sorted( zip(T,range(len(T))) ) ]
Tnew = [ T[i] for i in INDEX ]
Snew = [ S[i] for i in INDEX ]
T,S = Tnew,Snew

# a function for presenting numbers in a neat form
def neat(X):
	try:
		return [neat(x) for x in X]
	except:
		return round(10000*X)/10000.



###################
#                 #
# SIMULATION PART #
#                 #
###################

# converge 1st infrastructue
stop = 0.0000001
max_iter = 100000

results = []

for X in range(len(S)):
	step = 0.1
	if not parSilent: print '## Infrastructure',X,'##',T[X],'##'
	params = [.001,1.,[0.25 for i in range(len(S))]]
	oldp = params
	oldE = 1000
	impatience = 0
	for _iter_ in range(100000):
		#E = 100
		E = 0
		dE = [0.,0.,[0.]*len(S)]
		SW = 0
		
		for t in range(0,len(S[X])-1):
			w = min(1., .1 + (1-S[X][t])/(1-0.95*max(S[X])) )
			d = S[X][t+1] - S[X][t]
			f = params[0] * min(1.,1 - S[X][t])**params[1]
			for i in range(len(S)):
				f *= max(0.01,S[i][t])**params[2][i]
			E += w*(f-d)**2
			dE[0] += w*(f-d)*f/params[0]
			dE[1] += w*(f-d)*f*sc.log(max(0.01,1-S[X][t]) )
			for i in range(len(S)):
				dE[2][i] += w*(f-d)*f*sc.log(max(0.01,S[i][t]))
			SW += w
		
		E = E/SW
		
		if E > oldE:
			step = max( 0.8*step, 0.00001 )
			params = oldp # backup copy of params
			E = oldE
		elif oldE - E < stop:
			break
		else:
			step = step*1.1
			#impatience += 1
			#if impatience == 5:
			#	impatience = 0
			#	step = step*1.1
		
		if _iter_%100 == 0 and not parSilent:
			print ' ',_iter_, neat(E), neat(params) #neat(dE)
		
		oldp = [params[0],params[1],list(params[2])] # backup copy
		
		params[0] -= dE[0]/SW*step
		params[1] -= dE[1]/SW*step
		for i in range(len(S)):
			params[2][i] -= dE[2][i]/SW*step
		params[0] = max(0.0001,params[0])
		params[1] = max(0.0001,params[1])
		for i in range(len(S)):
			params[2][i] = max(0,params[2][i])
		
		oldE = E

	# time to compute the actual error rate

	if not parSilent: 
		print '  ***** convergence *****'
		print '  Avg squared error:', neat( E )
		print '  Standard deviation:',neat( E**0.5 )
		print '  ***** done *****'
	
	results.append( (params,E**0.5) )

# present the results
print
print '## ## ## RESULTS ## ## ##'
print
for X in range(len(S)):
	print 'Instrastructure',X,'('+T[X]+')'
	if parTest:
		mean = sum([z for z in S[X]])/len(S[X])
		var = sum([(z-mean)**2 for z in S[X]])/(len(S[X])-1)
		print '\t R^2 = ', int(10000*(1. - results[X][1]**2 / var ))/10000.
	elif parLaTeX:
		print
		print 'x_{'+str(X)+'}(t+1) = x_{'+str(X)+'}(t) + '
		print '\t',neat(results[X][0][0]),'\cdot (1-x_{'+str(X)+'}(t))^{',neat(results[X][0][1]),'}'
		for j in range(len(S)):
			print '\t \cdot x_{'+str(j)+'}(t)^{',neat(results[X][0][2][j]),'}'
		print '\t +',neat(results[X][1]),' \cdot \epsilon_{',X,'}(t)'
		print
	else:
		print '\t Alpha_i  =', neat(results[X][0][0])
		print '\t Beta_i   =', neat(results[X][0][1])
		print '\t Gamma_ij =', neat(results[X][0][2])
		print '\t Stdev_i  =', neat(results[X][1])
	print


def COLS(field):
	x = field.upper()
	if   x == 'POWER':	 return 'y'
	elif x == 'WATER':	 return 'b'
	elif x == 'GAS':	 return 'm'
	elif x == 'TELECOM': return 'r'
	elif x == 'HOSPITAL':	 return 'g'
	elif x == 'FIXED':	 return 'c'
	elif x == 'MOBILE':	 return 'r'
	else: return 'k'

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


#if parMField and not parStoch:

# mean field simulation
MF = [ [z[0]] for z in S ]
for t in range( len(S[0]) - 1 ):
	for i in range(len(S)):
		alpha = results[i][0][0]
		beta  = results[i][0][1]
		gamma = results[i][0][2]
		adv = alpha * (1 - MF[i][t])**beta
		for j in range(len(S)):
			if gamma[j]==0: continue
			adv *= ( MF[j][t] ** gamma[j] )
		newval = MF[i][t] + adv
		newval = max(0.01,newval)
		newval = min(1.00,newval)
		MF[i].append( newval )
# plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
plt.figure( figsize=FSIZE )
cols = 'brgcykm'
linst = ': : -. -- : -.'.split(' ')
for i in range(len(S)):
	plt.plot( S[i] , COLS(T[i]) + '-' , label = T[i]+', data') # cols[i%7]
for i in range(len(S)):
	plt.plot( MF[i] , COLS(T[i]) + linst[i%6] , label = T[i]+', MFA') # cols[i%7]
fz = 12 - (len(S)>3) - (len(S)>5)
plt.legend(loc=4,ncol=2,fontsize=fz)
plt.xlabel('Days, $t$')
plt.ylabel('Functionality, $x_i(t)$')
plt.tight_layout()
#plt.show()
plt.savefig('out1.pdf',bbox_inches='tight')

# mean field residuals
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
plt.figure( figsize=FSIZE )
cols = 'brgcykm'
for i in range(len(S)):
	(pX,pY) = ([],[])
	for t in range(0, min(len(S[i]),len(MF[i])) ):
		pX.append( S[i][t] )
		pY.append( (S[i][t]-MF[i][t]) )
	pP = sorted(zip(pX,pY))
	pX = [x for (x,y) in pP]
	pY = [y for (x,y) in pP]
	#plt.plot(pX,pY,estilo(T[i]),label = T[i],alpha=0.75)
	plt.plot(pX,pY,estilo(T[i]),label = T[i],alpha=0.75)
plt.plot([0,1],[0,0],'k-',alpha=0.5)
plt.legend(loc=4,ncol=2,fontsize=fz)
plt.xlabel('Functionality, $x_i(t)$')
plt.ylabel('Residuals, $x_i(t)-\~{x}_i(t)$')
plt.xlim(0,1)
plt.ylim(-1,1)
plt.tight_layout()
plt.savefig('res1.pdf',bbox_inches='tight')

# mean field residuals
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
plt.figure( figsize=FSIZE )
cols = 'brgcykm'
for i in range(len(S)):
	(pX,pY) = ([],[])
	for t in range(1, min(len(S[i]),len(MF[i])) ):
		pX.append( (MF[i][t]-MF[i][t-1]) )
		pY.append( (S[i][t]-MF[i][t]) )
	pP = sorted(zip(pX,pY))
	pX = [x for (x,y) in pP]
	pY = [y for (x,y) in pP]
	plt.plot(pX,pY,estilo(T[i]),label = T[i],alpha=0.75)
plt.plot([0,1],[0,0],'k-',alpha=0.5)
plt.legend(loc=4,ncol=2,fontsize=fz)
plt.xlabel('Restoration rate, $\phi_i(t)$')
plt.ylabel('Residuals, $x_i(t)-\~{x}_i(t)$')
plt.xlim(0,1)
plt.ylim(-1,1)
plt.tight_layout()
plt.savefig('res2.pdf',bbox_inches='tight')


#if parStoch and not parMField:

from random import gauss #normal distro
# stochastic simulations ...... 
M = [ [[] for z_ in z] for z in S ]
A = []
for _iter_ in range(600):
	R = [ [z[0]] for z in S ]
	for t in range( len(S[0]) - 1 ):
		for i in range(len(S)):
			alpha = results[i][0][0]
			beta  = results[i][0][1]
			gamma = results[i][0][2]
			adv = alpha * (1 - R[i][t])**beta
			for j in range(len(S)):
				if gamma[j]==0: continue
				adv *= ( R[j][t] ** gamma[j] )
			newval = R[i][t] + adv + results[i][1]*gauss(0,1)
			newval = max(0.01,newval)
			newval = min(1.00,newval)
			R[i].append( newval )
	for i in range(len(R)):
		for j in range(len(R[i])):
			M[i][j].append(R[i][j])
	A.append(R)
# quantile curves ...... 
Q = []
for i in range(len(M)):
	q = []
	for j in range(len(M[i])):
		M[i][j].sort()
		q.append( (M[i][j][299]+M[i][j][300])/2. )
	Q.append(q)
# plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
plt.figure( figsize=FSIZE)
cols = 'brgcykm'
linst = ': : -. -- : -.'.split(' ')
for i in range(len(S)):
	plt.plot( S[i] , COLS(T[i]) + '-' , label = T[i]+', data') # cols[i%7]
for i in range(len(S)):
	plt.plot( Q[i] , COLS(T[i]) + linst[i%6] , label = T[i]+', Median') # cols[i%7]
fz = 12 - (len(S)>3) - (len(S)>5)
plt.legend(loc=4,ncol=2,fontsize=fz)
for r in A[:30]:
	for i in range(len(S)):
		plt.plot( r[i] , COLS(T[i]) + 'x' , alpha=.08 ) # cols[i%7] 
		#plt.plot( r[i] , cols[i%7] + 's' + linst[i%6] , alpha=.05 )
plt.xlabel('Days, $t$')
plt.ylabel('Functionality, $x_i(t)$')
plt.tight_layout()
#plt.show()
plt.savefig('out2.pdf',bbox_inches='tight')



### Compare the mean field and the median?
import scipy.stats as st

V1 = []  # median vs mean-field approx
V2 = []  # natural variability
V3 = []  # MFA versus data
V4 = []  # Median versus data

for i in range(len(MF)):
	var1 = sum([(u-v)**2 for (u,v) in zip(Q[i],MF[i])])/len(MF[i])
	var2 = sum([st.tvar([m[j] for m in M[i]]) for j in range(len(MF[i]))])/len(MF[i])
	var3 = sum([(MF[i][j] - S[i][j])**2 for j in range(len(S[i]))])/len(S[i])
	var4 = sum([ (Q[i][j] - S[i][j])**2 for j in range(len(S[i]))])/len(S[i])
	#print i, var1, var2
	V1.append(var1)
	V2.append(var2)
	V3.append(var3)
	V4.append(var4)

print
print sum(V1)/len(V1), sum(V2)/len(V2)
print


SS = sum([ results[i][1]**2 for i in range(len(S)) ])/len(S)
print "<Stdev> & Median:MSE & MFA:MSE & MFA:MSAE & Sim.Var."
print SS, "&", sum(V4)/len(V4), "&", sum(V3)/len(V3), "&", sum(V1)/len(V1), "&", sum(V2)/len(V2)


