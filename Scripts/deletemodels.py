import os

def deleteModels(folder):

    metric_file =os.path.join(folder, os.path.split(folder)[-1]+'.out') 
    with open(metric_file) as m_f:
        head = " Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = "
        i = 0
        best_ap = 0.0
        best_epoch = 0
        for line in m_f:
            if line.startswith(head):
                line = line.replace(head,'')
                cur_ap = float(line)
                if cur_ap > best_ap: 
                    best_ap = cur_ap
                    best_epoch = i
                i += 1

    keys = [str(n) for n in range(0,i+1)]
    keys.remove(str(best_epoch))
    print(best_epoch, best_ap)

    for path in os.listdir(folder):
        if path.endswith('.pth') and path.replace('model_','').replace('.pth','') in keys:
            os.remove(os.path.join(folder, path))
        if path.endswith('.pth') and best_epoch == path.replace('model_','').replace('.pth',''):
            os.rename(os.path.join(folder, path), os.path.join(folder, "model_{}_{}.pth".format(best_epoch, best_ap)))


if __name__ == "__main__":
    folder = 'results/alltraintest-voc[16, 20]_pb[body]_Adam:1e-4_1e-5'
    deleteModels(folder)