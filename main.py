import numpy as np
import string

class NeuralNetwork:
    def __init__(self, n_features,n_classes,n_hidden):
        # Num of features
        self.d = n_features
        # Num of neurons in output layer
        self.n = n_classes
        # Num of neurons in hidden layers
        self.h = n_hidden
        ## Weights and biases from Input to Hidden layer
        # Creating weight matrix
        # 0.01 * np.random.randn(self.d, self.h)
        self.W1 = np.random.randn(self.d, self.h) * np.sqrt(2. / self.d)
        # Creating bias terms in a 1xh vector
        self.b1 = np.zeros((1,self.h))
        ## Weights and biases from Hidden to Output layer
        self.W2 = np.random.randn(self.h, self.n) * np.sqrt(2. / self.h)
        self.b2 = np.zeros((1,self.n))

    def forward_prop(self,x):
        """
        Forward propagation function
        """
        # (Input > Hidden) Multiply weight with values and adidng bias term b1
        z1 = np.dot(x,self.W1)+self.b1
        # Apply ReLu activation fucntion to z1, this function returns z1 if positive else returns 0 if z1 is negative.
        A1 = np.maximum(0,z1)

        # (Hidden > Output) Multiply weight with values and adding bias term b2
        z2 = np.dot(A1,self.W2)+self.b2

        # z1 & z2 are Logits, outputs before actiavtion
        #applying Softmax actiavtion function to z2, covnerts values into probabilities
        #A2 = np.exp(z2)
        #A2 = A2/np.sum(A2,axis=1,keepdims=True)

        # More stable Softmax activation
        z2 -= np.max(z2, axis=1, keepdims=True)
        A2 = np.exp(z2)
        A2 /= np.sum(A2, axis=1, keepdims=True)

        return A1, A2
    
    def ce_loss(self, y_true, y_pre_pro):
        """
        Computes Cross-Entropy loss

        :param self: The neural network
        :param y_true: True labels of data points
        :param y_pre_pro: Predicted probabikities
        """

        num_examples = y_true.shape[0]
        yij_pij = -np.log(y_pre_pro[range(num_examples),y_true])
        loss = np.sum(yij_pij)/num_examples
        return loss
    
    def back_prop(self,x,y,A1,A2):
        num_examples = y.shape[0]

        # Copy logits into dZ2
        dZ2 = A2
        dZ2[range(num_examples),y] -= 1
        # Normalise gradients
        dZ2 /= num_examples

        # Compute derivative of loss w/ respect to W2
        dW2 = np.dot(A1.T, dZ2)
        # Compute derivative of loss w/ respect to b2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        # COmpute derivative of loss w/ respect to A1
        dA1 = np.dot(dZ2, self.W2.T)
        # Compute gradient for ReLu
        dA1[dA1<0]==0
        # Compute gradient for z1
        dZ1=dA1
        # Compute gradient for W1
        dW1 = np.dot(x.T, dZ1)
        # Compute gradient for b2
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def train(self,x,y,reg,epochs,eta):
        num_examples = x.shape[0]

        # forward and backward prop for each epoch
        for i in range(epochs):
            # Forward prop
            A1, A2=self.forward_prop(x)

            # Calc loss
            loss = self.ce_loss(y, A2)
            # Calc regularisation loss
            reg_loss = 0.5*reg*np.sum(self.W1*self.W1) + 0.5*reg*np.sum(self.W2*self.W2)
            # Calc total loss
            total_loss = loss + reg_loss
            
            if i % 100 == 0:
                print(f"Iteration {i}: Loss {total_loss}")
            
            # Back prop
            dW1, db1, dW2, db2 = self.back_prop(x,y,A1,A2)

            # Add regularisation gradient contribution
            dW2 += reg * self.W2
            dW1 += reg * self.W1

            # Update params
            self.W1 += -eta*dW1
            self.W2 += -eta*dW2
            self.b1 += -eta*db1
            self.b2 += -eta*db2
        
        print(f"Finished Training")

    def predict(self, x):
        # Forward prop
        _, y_prediction = self.forward_prop(x)
        # Convert class probabilities into class labels
        y_pred = np.argmax(y_prediction, axis=1)

        return y_pred
    
    def export_parameters(self, filename):
        with open(filename, "w") as f:
            # Save shapes first for easier import
            f.write(f"{self.W1.shape}\n")
            np.savetxt(f, self.W1, fmt="%.6f")
            
            f.write(f"{self.b1.shape}\n")
            np.savetxt(f, self.b1, fmt="%.6f")
            
            f.write(f"{self.W2.shape}\n")
            np.savetxt(f, self.W2, fmt="%.6f")
            
            f.write(f"{self.b2.shape}\n")
            np.savetxt(f, self.b2, fmt="%.6f")
        
        print(f"Parameters exported to {filename}")

    def import_parameters(self, filename):
        with open(filename, "r") as f:
            # Read shapes
            W1_shape = eval(f.readline())
            W1 = np.loadtxt([next(f) for _ in range(W1_shape[0])])
            
            b1_shape = eval(f.readline())
            b1 = np.loadtxt([next(f) for _ in range(b1_shape[0])])
            
            W2_shape = eval(f.readline())
            W2 = np.loadtxt([next(f) for _ in range(W2_shape[0])])
            
            b2_shape = eval(f.readline())
            b2 = np.loadtxt([next(f) for _ in range(b2_shape[0])])
        
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        
        print(f"Parameters imported from {filename}")


neural_net = NeuralNetwork(n_features=5,n_classes=26,n_hidden=100)


# AI generated code, this data would be replaced with actual data from sensors and labeled manually
# could try to find a way to automate this, but data would still have to be created
def generate_training_data(fp = "./dataset.txt"):
    np.random.seed(42)
    num_categories = 26
    n_features = 5

    # Create 26 random 5D centers
    centers = np.random.rand(num_categories, n_features)

    # Labels A–Z
    labels_letters = list(string.ascii_uppercase)
    num_samples = 5000

    # Random 5D samples
    x = np.random.rand(num_samples, n_features)

    # Compute distances to each center
    distances = np.linalg.norm(
        x[:, np.newaxis, :] - centers[np.newaxis, :, :],
        axis=2
    )

    # Assign label of closest center
    y_numeric = np.argmin(distances, axis=1)

    # Optional: convert to letters
    y_letters = np.array([labels_letters[i] for i in y_numeric])

    # Make y_numeric a column
    y_col = y_numeric.reshape(-1, 1)

    # Combine x and y
    dataset = np.hstack([x, y_col])
    np.savetxt(fp, dataset, fmt="%.6f", delimiter=",", header="f1,f2,f3,f4,f5,label", comments="")

def load_training_data(fp="./dataset.txt"):
    loaded = np.loadtxt(fp, delimiter=",", skiprows=1)
    x_loaded = loaded[:, :-1]
    y_loaded = loaded[:, -1].astype(int)

    return x_loaded, y_loaded

def test_random_input(neural_net):
    sample = np.random.rand(1, 5)

    _, probs = neural_net.forward_prop(sample)
    pred_numeric = np.argmax(probs, axis=1)[0]
    confidence = probs[0, pred_numeric]

    label = string.ascii_uppercase[pred_numeric]

    print("Random Input:", sample)
    print("Predicted Category:", label)
    print("Confidence:", confidence)

    return sample, label, confidence


# just to time the training duration
import time
start = time.time()

generate_training_data()
# training data can be loaded/exported
points, labels = load_training_data()
neural_net.train(points, labels, reg=1e-2, epochs=10000, eta=0.1) # reg=1e-3
# once the model has been trained, parameters can be exported to give the same results without retraining (could also keep the parameters with the most accuracy)
neural_net.export_parameters("./params.txt")
print("Training finished in %.2f seconds" % ({time.time()-start}))
print("Training accuracy: %.4f" % (np.mean(np.array(neural_net.predict(points))==labels)))

# testing a random input
test_random_input(neural_net)
