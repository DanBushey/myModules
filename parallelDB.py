import pdb
import subprocess
import pandas as pd
import time
import ipyparallel as ipp

#pdb.set_trace()
#Ben Arthur helped with the call for pyton-parallel processing on the cluster

def startParallel(n):
    # starts n number of jobs on the cluster
    line1 = 'ipcontroller --ip=* &' 
    #line2 =  'qsub -t ' + '1-' + str(n) + ' -b y -j y -o ./ipengine.log -cwd -V ipengine'
    #line2 =  'qsub -pe batch 2 -t ' + '1-' + str(n) + ' -b y -j y -o ./ipengine.log -cwd -V ipengine'
    '''
    Call on bash$
    'ipcontroller --ip=* &'
    bsub -J myjob[1-3] -o ./ipengine.log ipengine
    '''
    line2 = bsub -n 16 -J "busheyd[1-1]" -o ./ipengine.log  -V ipengine
    #subprocess.check_call([line1], shell = True)
    #subprocess.check_call([line2], shell = True)
    subprocess.Popen([line1], shell = True)
    subprocess.Popen([line2], shell = True)
    #after running this script python script stops - calls and generates but does not continue


def deleteAllJobs(*arg):
    line1 = 'qdel -u busheyd'
    subprocess.check_call([line1], shell = True)
    
def detelePyJobs(*arg):
    line1 = 'qdel -u busheyd ipengine'
    subprocess.check_call([line1], shell = True)
    
def startLocal(n):
    line1 = 'ipcluster start -n ' + str(n) + '&'
    subprocess.check_call([line1], shell = True)

def stopLocal():
    line1 = 'ipcluster stop'
    subprocess.check_call([line1], shell = True)

def getQstats():
    p=subprocess.Popen('qstat', stdout=subprocess.PIPE)
    out=p.communicate()
    outsplit = out[0].split('\n')
    headers = outsplit[0].split(' ')
    headers = filter(None, headers)
    indexforheaders = {headers[0]: 0, headers[1]: 1, headers[2]: 2, headers[3]: 3,headers[4]: 4,headers[5]: 5,headers[6]: 6, headers[8]: 7,headers[10]: 8 }
    #create dictionaries to add to pandas dataframe
    dict1={head : [] for head in headers}
    for row in outsplit[2:]:
        if row:
            for ckey in indexforheaders.keys():
                row2 = row.split(' ')
                row3 = filter(None, row2)
                #row=filter(None, row.split(' '))
                dict1[ckey].append(row3[indexforheaders[ckey]])
            dict1['slots'].append('NA')
            dict1['queue'].append('NA')
    return pd.DataFrame(dict1)

def startLocalWait(n, attempts1):
	# n = number of engines
	# attempts = number of times to try to connect to engines
	startLocal(n)
	startTime = time.time()
	cattempts = 0
	dview = []
	while len(dview) < n and cattempts < attempts1:
		try:
			rc = ipp.Client()
			dview = rc[:]
		except:
			ctime = time.time() - startTime
			print('Waiting for Engines', ctime, 'seconds waiting')
			time.sleep(3)
	print('Linked to engines')
	return dview
	
def startParallelWait(n, attempts1):
	startParallel(n)
	startTime = time.time()
	cattempts = 0
	dview = []
	while len(dview) < n and cattempts < attempts1:
		try:
			rc = ipp.Client()
			dview = rc[:]
		except:
			ctime = time.time() - startTime
			print('Waiting for Engines', ctime, 'seconds waiting')
			time.sleep(3)
	print('Linked to engines')
	return dview


        


