# Super Resolution  Pytorch
* Please USE teamviewer to control lab's computer
<br><code><font color="red">Make sure the computer is free to use!!!  </font></code> 
<br><code>Account : 443347086 <br>Password: 954rfc</code><br>

* Move to filepath <br><code> $ cd \home\nbmedl\Wei\pytorch\DSPCP\HW3\part2</code>
* Active pytorch enviroment <br><code>$ source activate pytorch</code>
## Train part
* Training <br><code>$ python train.py --nFeat 16 --nResBlock 2 --nEpochs 15 --cuda</code>
> --nFeat : Numbers of feature map ( 16 or 64 )<br>
> --nResBlock : Numbers of Residual block ( 2 or 8 )<br>
> --nEpochs : Numbers of Epoch ( 15 or 100 )<br>
> --cuda : use GPU to speedup
## Test part
* testing nFeat = 16, nResBlock = 2, nEpochs = 15
<br><code>$ python test.py -–model ./model_pretrained/net_F16B2_epoch_15.pth 
--input_image ./image_test/LR_zebra_test.png 
--output_filename ./result/F16B2_zebra_test.png 
--compare_image ./ref/HR_zebra_test.png --cuda</code>
> --model : pretrain model<br>
> --input_image : test low resolution image<br>
> --output_filename : processed image<br>
> --compare_image : reference image<br>
> --cuda : useGPU speedup<br>
> <font color='red'>TODO : Change the input image to complete your homework</font>

* testing nFeat = 64, nResBlock = 8, nEpochs = 100
<br><code>$ python test.py -–model ./model_pretrained/net_F64B8_epoch_100.pth 
--input_image ./image_test/LR_zebra_test.png 
--output_filename ./result/F64B8_zebra_test.png 
--compare_image ./ref/HR_zebra_test.png --cuda</code>
> --model : pretrain model<br>
> --input_image : test low resolution image<br>
> --output_filename : processed image<br>
> --compare_image : reference image<br>
> --cuda : useGPU speedup<br>
> <font color='red'>TODO : Change the input image to complete your homework</font>
