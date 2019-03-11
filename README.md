# Advanced_language_models
Seq2Seq, Memory and Attention

##Seq2Seq

Seq2Seq is a Neural Network architecture that allows inputs and outputs
to have different lengths. In particular it is useful for translation.

For this task the data was obtained at http://www.manythings.org/anki/ where
the user can choose any file he wants. These files contain sentences translated from
english to the language the user selects.

A common architecture is as follows:

english_sentence --> sequence --> encoder -> decoder --> unsequence --> spanish sentence

The encoder is an RNN with fixed length were we input an english sequence. To
handle sequences with variable lengths it is necesary to use padding previously.

    #Encoder
    encoder_input = Input([MAX_ENCODER_LENGTH,])
    embedded_input = encoder_embeddings(encoder_input)
    encoder_lstm = LSTM(LATENT_DIM, return_state = True)
    __, h_encoder, c_encoder = encoder_lstm(embedded_input)

During training we use a special trick called Teacher Forcing in which not only 
do we indicate the correct answer at the output of the RNN cell, but also at the input. This
accelerates learning.

    #Decoder
    #We use the decoder_input for teaching.
    decoder_input = Input([MAX_DECODER_LENGTH,])
    embedded_decoder_input = decoder_embeddings(decoder_input) #(batch_size, max_sequence_length, embedding_size)
    decoder_lstm = LSTM(LATENT_DIM, return_state = True, return_sequences = True)
    decoder_output, h_decoder, c_decoder = decoder_lstm(embedded_decoder_input, initial_state = [h_encoder, c_encoder])
    dense = Dense(MAX_DECODER_WORDS, activation = 'softmax')
    output = dense(decoder_output)
    
To translate new sentences we must be careful of the decoder output because previously we didn't connect the output of a cell to its
input for the next timestep. So we create another model reusing the weights learned during
training.

First, we use the encoder as normal and obtain the hidden state:

    #During prediction, use the decoder to output the hidden state.
    encoder_prediction_model = Model(inputs = [encoder_input], outputs = [h_encoder, c_encoder])

Second, we create an RNN of length one reusing the decoder learned previously.

    #Once we get the hidden state, pass it to the decoder that along the decoder input will output word indexes.
    input_decoder2 = Input([1, ])
    input_h = Input([LATENT_DIM, ])
    input_c = Input([LATENT_DIM, ])
    embedded_decoder_input2 = decoder_embeddings(input_decoder2)
    o2, h2, c2 = decoder_lstm(embedded_decoder_input2, initial_state=[input_h, input_c])
    outputs2 = dense(o2)
    decoder_prediction_model = Model(inputs = [input_decoder2, input_h, input_c], outputs = [outputs2, h2, c2])
    
Finally use a loop to get the current word prediction and feeding it back to the cell along with
the updated hidden state to get the next word prediction.

    for i in range(MAX_SEQUENCE_LENGTH):
        probs, h_s, c_s = decoder_prediction_model.predict([decoder_input_, h_s, c_s])
        #Do not take into account the first column.
        word_idx = np.argmax(probs[0, 0, 1:]) + 1
        word = decoder_index2word[word_idx]
        #If end of sentence token, break;
        if word == '<eos>':
            break;
        
        decoder_input_[0, 0] = word_idx;
        decoder_sentence.append(word)
        
For the loss function we used categorical cross-entropy where we had to hot encode the targets 
(One hot encoding consumes a lot of memory, that's why we couldn't train on large sentences). 
After training during 100 epochs we tested the model.

    English sentence: It was fabulous. 
    Spanish translation: estaba fabuloso.
    
    English sentence: Call me. 
    Spanish translation: llamadme.
    
    English sentence: Are you mad? 
    Spanish translation: ¿estás loco?
    
    English sentence: What's the plan? 
    Spanish translation: ¿cuál es el plan?
    
    English sentence: I need answers. 
    Spanish translation: necesito respuestas.
    
We got pretty good results. However, due to memory limitations the model was trained
only with short sentences (0-10 words).