from random import uniform


# Make a prediction with weights
def classify(row, weights):
    sum = 0;
    #add the inputs times the weights
    for i in range(len(row)-1):
        sum += (row[i] * weights[i])
    #add the bias
    sum += weights[-1]
    #activation function(Binary Step Function)
    if sum >= 0:
        return 1
    else:
        return 0
 
#Estimate Perceptron weights using stochastic gradient descent
def train(train_data, n_epoch, l_rate=1):
    weights = [];
    #randomize weights
    for i in range(len(train_data[0])):
        weights.append(uniform(-1, 1));

    for i in range(n_epoch):
        success = 0;
        for j in range(len(train_data)):
            #find with actual and excpected
            error = train_data[j][-1]-classify(train_data[j], weights)
            if (error == 0):
                success += 1
            #update weights
            for k in range(len(train_data[0])-1):
                weights[k] = weights[k]+l_rate*(error*train_data[j][k])
            #update bias
            weights[-1] = weights[-1]+l_rate*error

        #ourput success(excpect it to be low this is very basic)
        percent = success/len(train_data)
        print("Epoch", i+1, "....", percent*100, "% correct")
