import matplotlib.pyplot as plt
import json

timings = json.load(open('build/timings.json'))

for device in timings:
    for imageSize in timings[device]:
        x_axis = timings[device][imageSize].keys()
        y_axis = timings[device][imageSize].values()

        plt.plot(x_axis, y_axis)
        plt.title(device + ' image dimensions = ' +
                  imageSize + 'x' + imageSize)
        if device == 'GPU':
            plt.xlabel('block dimensions')
        elif device == 'CPU':
            plt.xlabel('number of cores')
        plt.ylabel('time (ms)')
        plt.savefig('plots/' + device + ' ' + imageSize + '.png')
        plt.close()


def average(dict):
    return list(map(lambda key: sum(dict[key].values()) /
        len(dict[key].values()), dict.keys()))


for device in timings:
	x_axis = timings[device].keys()
	y_axis = average(timings[device])
	
	plt.plot(x_axis, y_axis)
	plt.title(device)
	plt.xlabel('image dimensions')
	plt.ylabel('average time (ms)')
	plt.savefig('plots/' + device + ' average over compute' + '.png')
	plt.close()

def average(dict):
	ret = [0] * len(timings[device][next(iter(timings[device]))].keys())
	for key in dict:
		for i, val in enumerate(dict[key].values()):
			ret[i] += val
	return ret

bestConfig = {}
for device in timings:
	bestConfig[device] = ''

for device in timings:
	x_axis = list(timings[device][next(iter(timings[device]))].keys())
	y_axis = average(timings[device])
	bestConfig[device] = x_axis[y_axis.index(min(y_axis))]

	plt.plot(x_axis, y_axis)
	plt.title(device)
	if device == 'GPU':
		plt.xlabel('block dimensions')
	elif device == 'CPU':
		plt.xlabel('number of cores')
	plt.ylabel('average time (ms)')
	plt.savefig('plots/' + device + ' average over image' + '.png')
	plt.close()


def extractBest(dict, best):
	ret = []
	for key in dict:
		ret.append(dict[key][best])
	return ret

bestGPU = extractBest(timings['GPU'], bestConfig['GPU'])
bestCPU = extractBest(timings['CPU'], bestConfig['CPU'])

x_axis = timings['CPU'].keys()
y_axis = []
for i in range(len(x_axis)):
	y_axis.append(bestCPU[i] / bestGPU[i])

plt.plot(x_axis, y_axis)
plt.title(device)
plt.xlabel('image dimensions')
plt.ylabel('speedup')
plt.savefig('plots/speedup' + '.png')
plt.close()

