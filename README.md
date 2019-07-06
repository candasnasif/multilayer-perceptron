- I used Python3 language for this assignment.
- My code can run with following command:
		python3 assignment4.py positive.txt negative.txt vectors.txt 75

- I used multilayer perceptron and tensorflow for this assignment.
- I used '10' for positive sentences' label and '01' for negative sentences' label.
- I used GradientDescentOptimizer for this assignment.
- My vectors' default dimensions are 200.
- I prefered ReLu activation function for this assignment.But I got higher accuracy when I used linear activation on out layer in multilayer perceptron. If you want to execute the code with Linear activation you should uncomment the line 31("output_layer = tf.matmul(hidden_layer_2, weights['out']) + biases['out']") and comment lines 28("output_layer = tf.add(tf.matmul(hidden_layer_2, weights['out']), biases['out'])") and 29("output_layer = tf.nn.relu(output_layer)").
- In our assignment, the learning rate is given us 0.001 default. For this learning rate I gave 15 to training eproch. Because I achieved highest accuracy for this learning rate.
- I printed the outputs in the console and 'output.txt' file with static name.
- Example output of the code:

		Epoch: 0001 cost= 152.00443915418114
		Epoch: 0002 cost= 30.622000756094593
		Epoch: 0003 cost= 17.520130658169855
		Epoch: 0004 cost= 3.9062384965076458
		Epoch: 0005 cost= 2.670056184132893
		Epoch: 0006 cost= 0.16042811321178535
		Epoch: 0007 cost= 1.203420534445081
		Epoch: 0008 cost= 3.2432935030728176e-07
		Epoch: 0009 cost= 3.2218150072940837e-07
		Epoch: 0010 cost= 3.211075759404716e-07
		Optimization Finished!

		Accuracy: 0.7894736842105263 
- First, I did optimization, and I displayed the logs per epoch step, then wrote accuracy as shown.
- My average accuracy is 0.55(%55) for default learning rate and training epoch 5.
- I explained the changes on learning rate and the training epochs and the accuracy results on following lines
---------------------------------------------------------------------------
When I used the ReLu activation function on out layer:
---------------------------------------------------------------------------

		Learning rate = 0.001
		Training eproch = 5
		output:

		Epoch: 0001 cost= 21.144341857705115
		Epoch: 0002 cost= 0.6806580440418144
		Epoch: 0003 cost= 0.6806580440418144
		Epoch: 0004 cost= 0.6806580440418144
		Epoch: 0005 cost= 0.6806580440418144
		Optimization Finished!
		Accuracy: 0.5789473684210527
---------------------------------------------------------------------------
		Learning rate = 0.0001
		Training eproch = 5
		output:

		Epoch: 0001 cost= 109.7710020982467
		Epoch: 0002 cost= 52.09832165814802
		Epoch: 0003 cost= 32.73714413585502
		Epoch: 0004 cost= 20.665172715681628
		Epoch: 0005 cost= 18.728232030072693
		Optimization Finished!
		Accuracy: 0.631578947368421
---------------------------------------------------------------------------
		Learning rate = 0.0001
		Training eproch = 20
		output:

		Epoch: 0001 cost= 26.189330341579712
		Epoch: 0002 cost= 6.060185513081571
		Epoch: 0003 cost= 1.0930882670857898
		Epoch: 0004 cost= 0.6789721669377512
		Epoch: 0005 cost= 0.6619243364076359
		Epoch: 0006 cost= 0.6619243364076359
		Epoch: 0007 cost= 0.6619243364076359
		Epoch: 0008 cost= 0.6619243364076359
		Epoch: 0009 cost= 0.6619243364076359
		Epoch: 0010 cost= 0.6619243364076359
		Epoch: 0011 cost= 0.6619243364076359
		Epoch: 0012 cost= 0.6619243364076359
		Epoch: 0013 cost= 0.6619243364076359
		Epoch: 0014 cost= 0.6619243364076359
		Epoch: 0015 cost= 0.6619243364076359
		Epoch: 0016 cost= 0.6619243364076359
		Epoch: 0017 cost= 0.6619243364076359
		Epoch: 0018 cost= 0.6619243364076359
		Epoch: 0019 cost= 0.6619243364076359
		Epoch: 0020 cost= 0.6619243364076359
		Optimization Finished!
		Accuracy: 0.5526315789473685
---------------------------------------------------------------------------
		Learning rate = 0.1
		Training eproch = 2
		output:

		Epoch: 0001 cost= 1.217675707556984
		Epoch: 0002 cost= 0.6931471824645984
		Optimization Finished!
		Accuracy: 0.39473684210526316
---------------------------------------------------------------------------
When I used the Linear activation function on out layer:
---------------------------------------------------------------------------

		Learning rate = 0.1
		Training eproch = 2
		output:

		Epoch: 0001 cost= 1.0228717111650002e+16
		Epoch: 0002 cost= 4.91555375149665e+34
		Optimization Finished!
		Accuracy: 0.5789473684210527

		Learning rate = 0.0001
		Training eproch = 20
		output:
---------------------------------------------------------------------------
		Epoch: 0001 cost= 133.1243749051471
		Epoch: 0002 cost= 91.61277197073166
		Epoch: 0003 cost= 71.3739060560482
		Epoch: 0004 cost= 54.88968921062613
		Epoch: 0005 cost= 45.438339961308515
		Epoch: 0006 cost= 31.18640113607159
		Epoch: 0007 cost= 24.824908807873612
		Epoch: 0008 cost= 20.94663226064316
		Epoch: 0009 cost= 19.82809093652641
		Epoch: 0010 cost= 13.563570987466756
		Epoch: 0011 cost= 13.153558221874881
		Epoch: 0012 cost= 9.983728487622162
		Epoch: 0013 cost= 9.158435246004798
		Epoch: 0014 cost= 6.053227093569413
		Epoch: 0015 cost= 5.141234482729601
		Epoch: 0016 cost= 3.6750488870290523
		Epoch: 0017 cost= 2.368318357531968
		Epoch: 0018 cost= 3.7367115845848806
		Epoch: 0019 cost= 2.176094612737785
		Epoch: 0020 cost= 1.8455332470798664
		Optimization Finished!
		Accuracy: 0.6842105263157895
---------------------------------------------------------------------------
		Learning rate = 0.0001
		Training eproch = 5
		output:

		Epoch: 0001 cost= 63.33287900511743
		Epoch: 0002 cost= 40.339910988413266
		Epoch: 0003 cost= 29.495862615779338
		Epoch: 0004 cost= 21.96345043826698
		Epoch: 0005 cost= 17.640762456713976
		Optimization Finished!
		Accuracy: 0.7368421052631579
---------------------------------------------------------------------------
		Learning rate = 0.001
		Training eproch = 5
		output:

		Epoch: 0001 cost= 217.55632358198767
		Epoch: 0002 cost= 39.151891193141836
		Epoch: 0003 cost= 5.884183071469775
		Epoch: 0004 cost= 11.952320030143605
		Epoch: 0005 cost= 4.894022317970011
		Optimization Finished!
		Accuracy: 0.7567567567567568


EXPLANATION: We can see clearly that we achieved higher accuracy when we used the Linear activation function on out layer. We can achieve %85 accuracy when we use Linear activation.

The accuracy rate for ReLu : %45 - %65
The accuracy rate for Linear : %55 - %85

I achieved the best accuracy when learning rate was 0.001, the training eproch was 5 and the out layer activation function was Linear.
