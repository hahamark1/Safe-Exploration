       ЃK"	   ЄnзAbrain.Event:2/,і"џ      ZЧЇТ	Уe/ЄnзA"ў

permute_1_inputPlaceholder*$
shape:џџџџџџџџџ(P*
dtype0*/
_output_shapes
:џџџџџџџџџ(P
q
permute_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:

permute_1/transpose	Transposepermute_1_inputpermute_1/transpose/perm*
T0*/
_output_shapes
:џџџџџџџџџ(P*
Tperm0
v
conv2d_1/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
`
conv2d_1/random_uniform/minConst*
valueB
 *ѓЕН*
dtype0*
_output_shapes
: 
`
conv2d_1/random_uniform/maxConst*
valueB
 *ѓЕ=*
dtype0*
_output_shapes
: 
В
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
seedБџх)*
T0*
dtype0*&
_output_shapes
:*
seed2ўЏ
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
_output_shapes
: *
T0

conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*
T0*&
_output_shapes
:

conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*
T0*&
_output_shapes
:

conv2d_1/kernel
VariableV2*
dtype0*&
_output_shapes
:*
	container *
shape:*
shared_name 
Ш
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:

conv2d_1/kernel/readIdentityconv2d_1/kernel*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_1/kernel
[
conv2d_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
conv2d_1/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
­
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
t
conv2d_1/bias/readIdentityconv2d_1/bias*
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:
s
"conv2d_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ђ
conv2d_1/convolutionConv2Dpermute_1/transposeconv2d_1/kernel/read*/
_output_shapes
:џџџџџџџџџ	*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID

conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ	*
T0
e
activation_1/ReluReluconv2d_1/BiasAdd*/
_output_shapes
:џџџџџџџџџ	*
T0
v
conv2d_2/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
`
conv2d_2/random_uniform/minConst*
valueB
 *   О*
dtype0*
_output_shapes
: 
`
conv2d_2/random_uniform/maxConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
В
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
T0*
dtype0*&
_output_shapes
:*
seed2эњ*
seedБџх)
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
T0*
_output_shapes
: 

conv2d_2/random_uniform/mulMul%conv2d_2/random_uniform/RandomUniformconv2d_2/random_uniform/sub*
T0*&
_output_shapes
:

conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*
T0*&
_output_shapes
:

conv2d_2/kernel
VariableV2*
shape:*
shared_name *
dtype0*&
_output_shapes
:*
	container 
Ш
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(

conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
[
conv2d_2/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
conv2d_2/bias
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
­
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:
t
conv2d_2/bias/readIdentityconv2d_2/bias*
_output_shapes
:*
T0* 
_class
loc:@conv2d_2/bias
s
"conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
№
conv2d_2/convolutionConv2Dactivation_1/Reluconv2d_2/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ

conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ*
T0
e
activation_2/ReluReluconv2d_2/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ
v
conv2d_3/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
`
conv2d_3/random_uniform/minConst*
_output_shapes
: *
valueB
 *:ЭО*
dtype0
`
conv2d_3/random_uniform/maxConst*
valueB
 *:Э>*
dtype0*
_output_shapes
: 
В
%conv2d_3/random_uniform/RandomUniformRandomUniformconv2d_3/random_uniform/shape*
dtype0*&
_output_shapes
:*
seed2ўк*
seedБџх)*
T0
}
conv2d_3/random_uniform/subSubconv2d_3/random_uniform/maxconv2d_3/random_uniform/min*
_output_shapes
: *
T0

conv2d_3/random_uniform/mulMul%conv2d_3/random_uniform/RandomUniformconv2d_3/random_uniform/sub*
T0*&
_output_shapes
:

conv2d_3/random_uniformAddconv2d_3/random_uniform/mulconv2d_3/random_uniform/min*&
_output_shapes
:*
T0

conv2d_3/kernel
VariableV2*
shared_name *
dtype0*&
_output_shapes
:*
	container *
shape:
Ш
conv2d_3/kernel/AssignAssignconv2d_3/kernelconv2d_3/random_uniform*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel

conv2d_3/kernel/readIdentityconv2d_3/kernel*
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:
[
conv2d_3/ConstConst*
dtype0*
_output_shapes
:*
valueB*    
y
conv2d_3/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
­
conv2d_3/bias/AssignAssignconv2d_3/biasconv2d_3/Const*
use_locking(*
T0* 
_class
loc:@conv2d_3/bias*
validate_shape(*
_output_shapes
:
t
conv2d_3/bias/readIdentityconv2d_3/bias*
T0* 
_class
loc:@conv2d_3/bias*
_output_shapes
:
s
"conv2d_3/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
№
conv2d_3/convolutionConv2Dactivation_2/Reluconv2d_3/kernel/read*
paddingVALID*/
_output_shapes
:џџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

conv2d_3/BiasAddBiasAddconv2d_3/convolutionconv2d_3/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ
e
activation_3/ReluReluconv2d_3/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ
`
flatten_1/ShapeShapeactivation_3/Relu*
T0*
out_type0*
_output_shapes
:
g
flatten_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
i
flatten_1/strided_slice/stack_1Const*
_output_shapes
:*
valueB: *
dtype0
i
flatten_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Џ
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
end_mask*
_output_shapes
:*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask 
Y
flatten_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
~
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
\
flatten_1/stack/0Const*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*
T0*

axis *
N*
_output_shapes
:

flatten_1/ReshapeReshapeactivation_3/Reluflatten_1/stack*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
m
dense_1/random_uniform/shapeConst*
valueB"`      *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *b'О*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *b'>*
dtype0*
_output_shapes
: 
Ј
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
seedБџх)*
T0*
dtype0*
_output_shapes
:	`*
seed2мпK
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
_output_shapes
: *
T0

dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes
:	`

dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes
:	`

dense_1/kernel
VariableV2*
dtype0*
_output_shapes
:	`*
	container *
shape:	`*
shared_name 
Н
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes
:	`
|
dense_1/kernel/readIdentitydense_1/kernel*
_output_shapes
:	`*
T0*!
_class
loc:@dense_1/kernel
\
dense_1/ConstConst*
_output_shapes	
:*
valueB*    *
dtype0
z
dense_1/bias
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Њ
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:
r
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:

dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 

dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
]
activation_4/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
m
dense_2/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *ЃЎXО*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *ЃЎX>*
dtype0*
_output_shapes
: 
Љ
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
T0*
dtype0*
_output_shapes
:	*
seed2цэ*
seedБџх)
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
_output_shapes
: *
T0

dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
_output_shapes
:	*
T0

dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
_output_shapes
:	*
T0

dense_2/kernel
VariableV2*
dtype0*
_output_shapes
:	*
	container *
shape:	*
shared_name 
Н
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes
:	
|
dense_2/kernel/readIdentitydense_2/kernel*
_output_shapes
:	*
T0*!
_class
loc:@dense_2/kernel
Z
dense_2/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_2/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Љ
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:

dense_2/MatMulMatMulactivation_4/Reludense_2/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
d
activation_5/IdentityIdentitydense_2/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
m
dense_3/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
_
dense_3/random_uniform/minConst*
valueB
 *ђъ-П*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ђъ-?
Ї
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
seedБџх)*
T0*
dtype0*
_output_shapes

:*
seed2јh
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
_output_shapes
: *
T0

dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
T0*
_output_shapes

:
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
_output_shapes

:*
T0

dense_3/kernel
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
М
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:
{
dense_3/kernel/readIdentitydense_3/kernel*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:
Z
dense_3/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_3/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Љ
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:
q
dense_3/bias/readIdentitydense_3/bias*
_output_shapes
:*
T0*
_class
loc:@dense_3/bias

dense_3/MatMulMatMuldense_2/BiasAdddense_3/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
m
lambda_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
o
lambda_1/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
o
lambda_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
Д
lambda_1/strided_sliceStridedSlicedense_3/BiasAddlambda_1/strided_slice/stacklambda_1/strided_slice/stack_1lambda_1/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*#
_output_shapes
:џџџџџџџџџ*
Index0*
T0*
shrink_axis_mask
b
lambda_1/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

lambda_1/ExpandDims
ExpandDimslambda_1/strided_slicelambda_1/ExpandDims/dim*'
_output_shapes
:џџџџџџџџџ*

Tdim0*
T0
o
lambda_1/strided_slice_1/stackConst*
valueB"       *
dtype0*
_output_shapes
:
q
 lambda_1/strided_slice_1/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
q
 lambda_1/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Р
lambda_1/strided_slice_1StridedSlicedense_3/BiasAddlambda_1/strided_slice_1/stack lambda_1/strided_slice_1/stack_1 lambda_1/strided_slice_1/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:џџџџџџџџџ*
T0*
Index0
t
lambda_1/addAddlambda_1/ExpandDimslambda_1/strided_slice_1*'
_output_shapes
:џџџџџџџџџ*
T0
o
lambda_1/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB"       
q
 lambda_1/strided_slice_2/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
q
 lambda_1/strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Р
lambda_1/strided_slice_2StridedSlicedense_3/BiasAddlambda_1/strided_slice_2/stack lambda_1/strided_slice_2/stack_1 lambda_1/strided_slice_2/stack_2*'
_output_shapes
:џџџџџџџџџ*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
_
lambda_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:

lambda_1/MeanMeanlambda_1/strided_slice_2lambda_1/Const*
_output_shapes

:*

Tidx0*
	keep_dims(*
T0
b
lambda_1/subSublambda_1/addlambda_1/Mean*
T0*'
_output_shapes
:џџџџџџџџџ
_
Adam/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
s
Adam/iterations
VariableV2*
shared_name *
dtype0	*
_output_shapes
: *
	container *
shape: 
О
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
use_locking(*
T0	*"
_class
loc:@Adam/iterations*
validate_shape(*
_output_shapes
: 
v
Adam/iterations/readIdentityAdam/iterations*
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: 
Z
Adam/lr/initial_valueConst*
valueB
 *o9*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 

Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
use_locking(*
T0*
_class
loc:@Adam/lr*
validate_shape(*
_output_shapes
: 
^
Adam/lr/readIdentityAdam/lr*
T0*
_class
loc:@Adam/lr*
_output_shapes
: 
^
Adam/beta_1/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
o
Adam/beta_1
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
Ў
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
use_locking(*
T0*
_class
loc:@Adam/beta_1*
validate_shape(*
_output_shapes
: 
j
Adam/beta_1/readIdentityAdam/beta_1*
T0*
_class
loc:@Adam/beta_1*
_output_shapes
: 
^
Adam/beta_2/initial_valueConst*
valueB
 *wО?*
dtype0*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Ў
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/beta_2
j
Adam/beta_2/readIdentityAdam/beta_2*
_output_shapes
: *
T0*
_class
loc:@Adam/beta_2
]
Adam/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n

Adam/decay
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
Њ
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
use_locking(*
T0*
_class
loc:@Adam/decay*
validate_shape(*
_output_shapes
: 
g
Adam/decay/readIdentity
Adam/decay*
_output_shapes
: *
T0*
_class
loc:@Adam/decay

permute_1_input_1Placeholder*/
_output_shapes
:џџџџџџџџџ(P*$
shape:џџџџџџџџџ(P*
dtype0
s
permute_1_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:

permute_1_1/transpose	Transposepermute_1_input_1permute_1_1/transpose/perm*
Tperm0*
T0*/
_output_shapes
:џџџџџџџџџ(P
x
conv2d_1_1/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
b
conv2d_1_1/random_uniform/minConst*
valueB
 *ѓЕН*
dtype0*
_output_shapes
: 
b
conv2d_1_1/random_uniform/maxConst*
valueB
 *ѓЕ=*
dtype0*
_output_shapes
: 
Ж
'conv2d_1_1/random_uniform/RandomUniformRandomUniformconv2d_1_1/random_uniform/shape*
dtype0*&
_output_shapes
:*
seed2гММ*
seedБџх)*
T0

conv2d_1_1/random_uniform/subSubconv2d_1_1/random_uniform/maxconv2d_1_1/random_uniform/min*
_output_shapes
: *
T0

conv2d_1_1/random_uniform/mulMul'conv2d_1_1/random_uniform/RandomUniformconv2d_1_1/random_uniform/sub*
T0*&
_output_shapes
:

conv2d_1_1/random_uniformAddconv2d_1_1/random_uniform/mulconv2d_1_1/random_uniform/min*&
_output_shapes
:*
T0

conv2d_1_1/kernel
VariableV2*
dtype0*&
_output_shapes
:*
	container *
shape:*
shared_name 
а
conv2d_1_1/kernel/AssignAssignconv2d_1_1/kernelconv2d_1_1/random_uniform*
use_locking(*
T0*$
_class
loc:@conv2d_1_1/kernel*
validate_shape(*&
_output_shapes
:

conv2d_1_1/kernel/readIdentityconv2d_1_1/kernel*
T0*$
_class
loc:@conv2d_1_1/kernel*&
_output_shapes
:
]
conv2d_1_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
{
conv2d_1_1/bias
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
Е
conv2d_1_1/bias/AssignAssignconv2d_1_1/biasconv2d_1_1/Const*"
_class
loc:@conv2d_1_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
z
conv2d_1_1/bias/readIdentityconv2d_1_1/bias*
T0*"
_class
loc:@conv2d_1_1/bias*
_output_shapes
:
u
$conv2d_1_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ј
conv2d_1_1/convolutionConv2Dpermute_1_1/transposeconv2d_1_1/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ	

conv2d_1_1/BiasAddBiasAddconv2d_1_1/convolutionconv2d_1_1/bias/read*/
_output_shapes
:џџџџџџџџџ	*
T0*
data_formatNHWC
i
activation_1_1/ReluReluconv2d_1_1/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ	
x
conv2d_2_1/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
b
conv2d_2_1/random_uniform/minConst*
_output_shapes
: *
valueB
 *   О*
dtype0
b
conv2d_2_1/random_uniform/maxConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
Ж
'conv2d_2_1/random_uniform/RandomUniformRandomUniformconv2d_2_1/random_uniform/shape*
seedБџх)*
T0*
dtype0*&
_output_shapes
:*
seed2ћшТ

conv2d_2_1/random_uniform/subSubconv2d_2_1/random_uniform/maxconv2d_2_1/random_uniform/min*
_output_shapes
: *
T0

conv2d_2_1/random_uniform/mulMul'conv2d_2_1/random_uniform/RandomUniformconv2d_2_1/random_uniform/sub*
T0*&
_output_shapes
:

conv2d_2_1/random_uniformAddconv2d_2_1/random_uniform/mulconv2d_2_1/random_uniform/min*&
_output_shapes
:*
T0

conv2d_2_1/kernel
VariableV2*
shape:*
shared_name *
dtype0*&
_output_shapes
:*
	container 
а
conv2d_2_1/kernel/AssignAssignconv2d_2_1/kernelconv2d_2_1/random_uniform*
use_locking(*
T0*$
_class
loc:@conv2d_2_1/kernel*
validate_shape(*&
_output_shapes
:

conv2d_2_1/kernel/readIdentityconv2d_2_1/kernel*&
_output_shapes
:*
T0*$
_class
loc:@conv2d_2_1/kernel
]
conv2d_2_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
{
conv2d_2_1/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Е
conv2d_2_1/bias/AssignAssignconv2d_2_1/biasconv2d_2_1/Const*
use_locking(*
T0*"
_class
loc:@conv2d_2_1/bias*
validate_shape(*
_output_shapes
:
z
conv2d_2_1/bias/readIdentityconv2d_2_1/bias*
T0*"
_class
loc:@conv2d_2_1/bias*
_output_shapes
:
u
$conv2d_2_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
і
conv2d_2_1/convolutionConv2Dactivation_1_1/Reluconv2d_2_1/kernel/read*
paddingVALID*/
_output_shapes
:џџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

conv2d_2_1/BiasAddBiasAddconv2d_2_1/convolutionconv2d_2_1/bias/read*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ*
T0
i
activation_2_1/ReluReluconv2d_2_1/BiasAdd*/
_output_shapes
:џџџџџџџџџ*
T0
x
conv2d_3_1/random_uniform/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0
b
conv2d_3_1/random_uniform/minConst*
_output_shapes
: *
valueB
 *:ЭО*
dtype0
b
conv2d_3_1/random_uniform/maxConst*
valueB
 *:Э>*
dtype0*
_output_shapes
: 
Е
'conv2d_3_1/random_uniform/RandomUniformRandomUniformconv2d_3_1/random_uniform/shape*
dtype0*&
_output_shapes
:*
seed2жЙM*
seedБџх)*
T0

conv2d_3_1/random_uniform/subSubconv2d_3_1/random_uniform/maxconv2d_3_1/random_uniform/min*
T0*
_output_shapes
: 

conv2d_3_1/random_uniform/mulMul'conv2d_3_1/random_uniform/RandomUniformconv2d_3_1/random_uniform/sub*
T0*&
_output_shapes
:

conv2d_3_1/random_uniformAddconv2d_3_1/random_uniform/mulconv2d_3_1/random_uniform/min*
T0*&
_output_shapes
:

conv2d_3_1/kernel
VariableV2*
dtype0*&
_output_shapes
:*
	container *
shape:*
shared_name 
а
conv2d_3_1/kernel/AssignAssignconv2d_3_1/kernelconv2d_3_1/random_uniform*$
_class
loc:@conv2d_3_1/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0

conv2d_3_1/kernel/readIdentityconv2d_3_1/kernel*&
_output_shapes
:*
T0*$
_class
loc:@conv2d_3_1/kernel
]
conv2d_3_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
{
conv2d_3_1/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Е
conv2d_3_1/bias/AssignAssignconv2d_3_1/biasconv2d_3_1/Const*
use_locking(*
T0*"
_class
loc:@conv2d_3_1/bias*
validate_shape(*
_output_shapes
:
z
conv2d_3_1/bias/readIdentityconv2d_3_1/bias*
T0*"
_class
loc:@conv2d_3_1/bias*
_output_shapes
:
u
$conv2d_3_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
і
conv2d_3_1/convolutionConv2Dactivation_2_1/Reluconv2d_3_1/kernel/read*
paddingVALID*/
_output_shapes
:џџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

conv2d_3_1/BiasAddBiasAddconv2d_3_1/convolutionconv2d_3_1/bias/read*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ*
T0
i
activation_3_1/ReluReluconv2d_3_1/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ
d
flatten_1_1/ShapeShapeactivation_3_1/Relu*
_output_shapes
:*
T0*
out_type0
i
flatten_1_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
k
!flatten_1_1/strided_slice/stack_1Const*
_output_shapes
:*
valueB: *
dtype0
k
!flatten_1_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Й
flatten_1_1/strided_sliceStridedSliceflatten_1_1/Shapeflatten_1_1/strided_slice/stack!flatten_1_1/strided_slice/stack_1!flatten_1_1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
[
flatten_1_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:

flatten_1_1/ProdProdflatten_1_1/strided_sliceflatten_1_1/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
^
flatten_1_1/stack/0Const*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
z
flatten_1_1/stackPackflatten_1_1/stack/0flatten_1_1/Prod*
T0*

axis *
N*
_output_shapes
:

flatten_1_1/ReshapeReshapeactivation_3_1/Reluflatten_1_1/stack*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
o
dense_1_1/random_uniform/shapeConst*
valueB"`      *
dtype0*
_output_shapes
:
a
dense_1_1/random_uniform/minConst*
valueB
 *b'О*
dtype0*
_output_shapes
: 
a
dense_1_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *b'>
­
&dense_1_1/random_uniform/RandomUniformRandomUniformdense_1_1/random_uniform/shape*
dtype0*
_output_shapes
:	`*
seed2с­Щ*
seedБџх)*
T0

dense_1_1/random_uniform/subSubdense_1_1/random_uniform/maxdense_1_1/random_uniform/min*
T0*
_output_shapes
: 

dense_1_1/random_uniform/mulMul&dense_1_1/random_uniform/RandomUniformdense_1_1/random_uniform/sub*
_output_shapes
:	`*
T0

dense_1_1/random_uniformAdddense_1_1/random_uniform/muldense_1_1/random_uniform/min*
_output_shapes
:	`*
T0

dense_1_1/kernel
VariableV2*
dtype0*
_output_shapes
:	`*
	container *
shape:	`*
shared_name 
Х
dense_1_1/kernel/AssignAssigndense_1_1/kerneldense_1_1/random_uniform*
validate_shape(*
_output_shapes
:	`*
use_locking(*
T0*#
_class
loc:@dense_1_1/kernel

dense_1_1/kernel/readIdentitydense_1_1/kernel*
T0*#
_class
loc:@dense_1_1/kernel*
_output_shapes
:	`
^
dense_1_1/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    
|
dense_1_1/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
В
dense_1_1/bias/AssignAssigndense_1_1/biasdense_1_1/Const*
use_locking(*
T0*!
_class
loc:@dense_1_1/bias*
validate_shape(*
_output_shapes	
:
x
dense_1_1/bias/readIdentitydense_1_1/bias*
T0*!
_class
loc:@dense_1_1/bias*
_output_shapes	
:

dense_1_1/MatMulMatMulflatten_1_1/Reshapedense_1_1/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

dense_1_1/BiasAddBiasAdddense_1_1/MatMuldense_1_1/bias/read*(
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
a
activation_4_1/ReluReludense_1_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
o
dense_2_1/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
a
dense_2_1/random_uniform/minConst*
valueB
 *ЃЎXО*
dtype0*
_output_shapes
: 
a
dense_2_1/random_uniform/maxConst*
valueB
 *ЃЎX>*
dtype0*
_output_shapes
: 
­
&dense_2_1/random_uniform/RandomUniformRandomUniformdense_2_1/random_uniform/shape*
T0*
dtype0*
_output_shapes
:	*
seed2рФ*
seedБџх)

dense_2_1/random_uniform/subSubdense_2_1/random_uniform/maxdense_2_1/random_uniform/min*
_output_shapes
: *
T0

dense_2_1/random_uniform/mulMul&dense_2_1/random_uniform/RandomUniformdense_2_1/random_uniform/sub*
_output_shapes
:	*
T0

dense_2_1/random_uniformAdddense_2_1/random_uniform/muldense_2_1/random_uniform/min*
T0*
_output_shapes
:	

dense_2_1/kernel
VariableV2*
dtype0*
_output_shapes
:	*
	container *
shape:	*
shared_name 
Х
dense_2_1/kernel/AssignAssigndense_2_1/kerneldense_2_1/random_uniform*
use_locking(*
T0*#
_class
loc:@dense_2_1/kernel*
validate_shape(*
_output_shapes
:	

dense_2_1/kernel/readIdentitydense_2_1/kernel*
_output_shapes
:	*
T0*#
_class
loc:@dense_2_1/kernel
\
dense_2_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
z
dense_2_1/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Б
dense_2_1/bias/AssignAssigndense_2_1/biasdense_2_1/Const*
use_locking(*
T0*!
_class
loc:@dense_2_1/bias*
validate_shape(*
_output_shapes
:
w
dense_2_1/bias/readIdentitydense_2_1/bias*
_output_shapes
:*
T0*!
_class
loc:@dense_2_1/bias

dense_2_1/MatMulMatMulactivation_4_1/Reludense_2_1/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

dense_2_1/BiasAddBiasAdddense_2_1/MatMuldense_2_1/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
o
dense_3_1/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
a
dense_3_1/random_uniform/minConst*
valueB
 *ђъ-П*
dtype0*
_output_shapes
: 
a
dense_3_1/random_uniform/maxConst*
_output_shapes
: *
valueB
 *ђъ-?*
dtype0
Ќ
&dense_3_1/random_uniform/RandomUniformRandomUniformdense_3_1/random_uniform/shape*
dtype0*
_output_shapes

:*
seed2ёА*
seedБџх)*
T0

dense_3_1/random_uniform/subSubdense_3_1/random_uniform/maxdense_3_1/random_uniform/min*
T0*
_output_shapes
: 

dense_3_1/random_uniform/mulMul&dense_3_1/random_uniform/RandomUniformdense_3_1/random_uniform/sub*
T0*
_output_shapes

:

dense_3_1/random_uniformAdddense_3_1/random_uniform/muldense_3_1/random_uniform/min*
_output_shapes

:*
T0

dense_3_1/kernel
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
Ф
dense_3_1/kernel/AssignAssigndense_3_1/kerneldense_3_1/random_uniform*
use_locking(*
T0*#
_class
loc:@dense_3_1/kernel*
validate_shape(*
_output_shapes

:

dense_3_1/kernel/readIdentitydense_3_1/kernel*
T0*#
_class
loc:@dense_3_1/kernel*
_output_shapes

:
\
dense_3_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
z
dense_3_1/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Б
dense_3_1/bias/AssignAssigndense_3_1/biasdense_3_1/Const*
use_locking(*
T0*!
_class
loc:@dense_3_1/bias*
validate_shape(*
_output_shapes
:
w
dense_3_1/bias/readIdentitydense_3_1/bias*
_output_shapes
:*
T0*!
_class
loc:@dense_3_1/bias

dense_3_1/MatMulMatMuldense_2_1/BiasAdddense_3_1/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

dense_3_1/BiasAddBiasAdddense_3_1/MatMuldense_3_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
o
lambda_1_1/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
q
 lambda_1_1/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
q
 lambda_1_1/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
О
lambda_1_1/strided_sliceStridedSlicedense_3_1/BiasAddlambda_1_1/strided_slice/stack lambda_1_1/strided_slice/stack_1 lambda_1_1/strided_slice/stack_2*#
_output_shapes
:џџџџџџџџџ*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
d
lambda_1_1/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

lambda_1_1/ExpandDims
ExpandDimslambda_1_1/strided_slicelambda_1_1/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:џџџџџџџџџ
q
 lambda_1_1/strided_slice_1/stackConst*
valueB"       *
dtype0*
_output_shapes
:
s
"lambda_1_1/strided_slice_1/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
s
"lambda_1_1/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
Ъ
lambda_1_1/strided_slice_1StridedSlicedense_3_1/BiasAdd lambda_1_1/strided_slice_1/stack"lambda_1_1/strided_slice_1/stack_1"lambda_1_1/strided_slice_1/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*'
_output_shapes
:џџџџџџџџџ*
Index0*
T0
z
lambda_1_1/addAddlambda_1_1/ExpandDimslambda_1_1/strided_slice_1*'
_output_shapes
:џџџџџџџџџ*
T0
q
 lambda_1_1/strided_slice_2/stackConst*
valueB"       *
dtype0*
_output_shapes
:
s
"lambda_1_1/strided_slice_2/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
s
"lambda_1_1/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
Ъ
lambda_1_1/strided_slice_2StridedSlicedense_3_1/BiasAdd lambda_1_1/strided_slice_2/stack"lambda_1_1/strided_slice_2/stack_1"lambda_1_1/strided_slice_2/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:џџџџџџџџџ*
Index0*
T0
a
lambda_1_1/ConstConst*
dtype0*
_output_shapes
:*
valueB"       

lambda_1_1/MeanMeanlambda_1_1/strided_slice_2lambda_1_1/Const*
T0*
_output_shapes

:*

Tidx0*
	keep_dims(
h
lambda_1_1/subSublambda_1_1/addlambda_1_1/Mean*
T0*'
_output_shapes
:џџџџџџџџџ

IsVariableInitializedIsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_1IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_2IsVariableInitializedconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_3IsVariableInitializedconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_4IsVariableInitializedconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_5IsVariableInitializedconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_6IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_7IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializeddense_2/kernel*
_output_shapes
: *!
_class
loc:@dense_2/kernel*
dtype0

IsVariableInitialized_9IsVariableInitializeddense_2/bias*
_output_shapes
: *
_class
loc:@dense_2/bias*
dtype0

IsVariableInitialized_10IsVariableInitializeddense_3/kernel*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_11IsVariableInitializeddense_3/bias*
dtype0*
_output_shapes
: *
_class
loc:@dense_3/bias

IsVariableInitialized_12IsVariableInitializedAdam/iterations*
_output_shapes
: *"
_class
loc:@Adam/iterations*
dtype0	
{
IsVariableInitialized_13IsVariableInitializedAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 

IsVariableInitialized_14IsVariableInitializedAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_15IsVariableInitializedAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_16IsVariableInitialized
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 

IsVariableInitialized_17IsVariableInitializedconv2d_1_1/kernel*$
_class
loc:@conv2d_1_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_18IsVariableInitializedconv2d_1_1/bias*"
_class
loc:@conv2d_1_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_19IsVariableInitializedconv2d_2_1/kernel*$
_class
loc:@conv2d_2_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_20IsVariableInitializedconv2d_2_1/bias*"
_class
loc:@conv2d_2_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_21IsVariableInitializedconv2d_3_1/kernel*
dtype0*
_output_shapes
: *$
_class
loc:@conv2d_3_1/kernel

IsVariableInitialized_22IsVariableInitializedconv2d_3_1/bias*"
_class
loc:@conv2d_3_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_23IsVariableInitializeddense_1_1/kernel*
dtype0*
_output_shapes
: *#
_class
loc:@dense_1_1/kernel

IsVariableInitialized_24IsVariableInitializeddense_1_1/bias*
_output_shapes
: *!
_class
loc:@dense_1_1/bias*
dtype0

IsVariableInitialized_25IsVariableInitializeddense_2_1/kernel*#
_class
loc:@dense_2_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_26IsVariableInitializeddense_2_1/bias*!
_class
loc:@dense_2_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_27IsVariableInitializeddense_3_1/kernel*#
_class
loc:@dense_3_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_28IsVariableInitializeddense_3_1/bias*!
_class
loc:@dense_3_1/bias*
dtype0*
_output_shapes
: 
Р
initNoOp^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^conv2d_2/kernel/Assign^conv2d_2/bias/Assign^conv2d_3/kernel/Assign^conv2d_3/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^Adam/iterations/Assign^Adam/lr/Assign^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^conv2d_1_1/kernel/Assign^conv2d_1_1/bias/Assign^conv2d_2_1/kernel/Assign^conv2d_2_1/bias/Assign^conv2d_3_1/kernel/Assign^conv2d_3_1/bias/Assign^dense_1_1/kernel/Assign^dense_1_1/bias/Assign^dense_2_1/kernel/Assign^dense_2_1/bias/Assign^dense_3_1/kernel/Assign^dense_3_1/bias/Assign
l
PlaceholderPlaceholder*
dtype0*&
_output_shapes
:*
shape:
А
AssignAssignconv2d_1_1/kernelPlaceholder*
use_locking( *
T0*$
_class
loc:@conv2d_1_1/kernel*
validate_shape(*&
_output_shapes
:
V
Placeholder_1Placeholder*
dtype0*
_output_shapes
:*
shape:
Є
Assign_1Assignconv2d_1_1/biasPlaceholder_1*
use_locking( *
T0*"
_class
loc:@conv2d_1_1/bias*
validate_shape(*
_output_shapes
:
n
Placeholder_2Placeholder*
dtype0*&
_output_shapes
:*
shape:
Д
Assign_2Assignconv2d_2_1/kernelPlaceholder_2*
use_locking( *
T0*$
_class
loc:@conv2d_2_1/kernel*
validate_shape(*&
_output_shapes
:
V
Placeholder_3Placeholder*
dtype0*
_output_shapes
:*
shape:
Є
Assign_3Assignconv2d_2_1/biasPlaceholder_3*
use_locking( *
T0*"
_class
loc:@conv2d_2_1/bias*
validate_shape(*
_output_shapes
:
n
Placeholder_4Placeholder*&
_output_shapes
:*
shape:*
dtype0
Д
Assign_4Assignconv2d_3_1/kernelPlaceholder_4*
T0*$
_class
loc:@conv2d_3_1/kernel*
validate_shape(*&
_output_shapes
:*
use_locking( 
V
Placeholder_5Placeholder*
dtype0*
_output_shapes
:*
shape:
Є
Assign_5Assignconv2d_3_1/biasPlaceholder_5*
use_locking( *
T0*"
_class
loc:@conv2d_3_1/bias*
validate_shape(*
_output_shapes
:
`
Placeholder_6Placeholder*
dtype0*
_output_shapes
:	`*
shape:	`
Ћ
Assign_6Assigndense_1_1/kernelPlaceholder_6*#
_class
loc:@dense_1_1/kernel*
validate_shape(*
_output_shapes
:	`*
use_locking( *
T0
X
Placeholder_7Placeholder*
dtype0*
_output_shapes	
:*
shape:
Ѓ
Assign_7Assigndense_1_1/biasPlaceholder_7*
_output_shapes	
:*
use_locking( *
T0*!
_class
loc:@dense_1_1/bias*
validate_shape(
`
Placeholder_8Placeholder*
dtype0*
_output_shapes
:	*
shape:	
Ћ
Assign_8Assigndense_2_1/kernelPlaceholder_8*#
_class
loc:@dense_2_1/kernel*
validate_shape(*
_output_shapes
:	*
use_locking( *
T0
V
Placeholder_9Placeholder*
shape:*
dtype0*
_output_shapes
:
Ђ
Assign_9Assigndense_2_1/biasPlaceholder_9*
_output_shapes
:*
use_locking( *
T0*!
_class
loc:@dense_2_1/bias*
validate_shape(
_
Placeholder_10Placeholder*
shape
:*
dtype0*
_output_shapes

:
Ќ
	Assign_10Assigndense_3_1/kernelPlaceholder_10*#
_class
loc:@dense_3_1/kernel*
validate_shape(*
_output_shapes

:*
use_locking( *
T0
W
Placeholder_11Placeholder*
dtype0*
_output_shapes
:*
shape:
Є
	Assign_11Assigndense_3_1/biasPlaceholder_11*
use_locking( *
T0*!
_class
loc:@dense_3_1/bias*
validate_shape(*
_output_shapes
:
^
SGD/iterations/initial_valueConst*
dtype0	*
_output_shapes
: *
value	B	 R 
r
SGD/iterations
VariableV2*
shared_name *
dtype0	*
_output_shapes
: *
	container *
shape: 
К
SGD/iterations/AssignAssignSGD/iterationsSGD/iterations/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*!
_class
loc:@SGD/iterations
s
SGD/iterations/readIdentitySGD/iterations*
T0	*!
_class
loc:@SGD/iterations*
_output_shapes
: 
Y
SGD/lr/initial_valueConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
j
SGD/lr
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 

SGD/lr/AssignAssignSGD/lrSGD/lr/initial_value*
use_locking(*
T0*
_class
loc:@SGD/lr*
validate_shape(*
_output_shapes
: 
[
SGD/lr/readIdentitySGD/lr*
T0*
_class
loc:@SGD/lr*
_output_shapes
: 
_
SGD/momentum/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
p
SGD/momentum
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
В
SGD/momentum/AssignAssignSGD/momentumSGD/momentum/initial_value*
use_locking(*
T0*
_class
loc:@SGD/momentum*
validate_shape(*
_output_shapes
: 
m
SGD/momentum/readIdentitySGD/momentum*
_output_shapes
: *
T0*
_class
loc:@SGD/momentum
\
SGD/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
m
	SGD/decay
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
І
SGD/decay/AssignAssign	SGD/decaySGD/decay/initial_value*
_class
loc:@SGD/decay*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
d
SGD/decay/readIdentity	SGD/decay*
T0*
_class
loc:@SGD/decay*
_output_shapes
: 

lambda_1_targetPlaceholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
r
lambda_1_sample_weightsPlaceholder*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ*
dtype0
p
loss/lambda_1_loss/subSublambda_1_1/sublambda_1_target*'
_output_shapes
:џџџџџџџџџ*
T0
m
loss/lambda_1_loss/SquareSquareloss/lambda_1_loss/sub*'
_output_shapes
:џџџџџџџџџ*
T0
t
)loss/lambda_1_loss/Mean/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
А
loss/lambda_1_loss/MeanMeanloss/lambda_1_loss/Square)loss/lambda_1_loss/Mean/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims( 
n
+loss/lambda_1_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
В
loss/lambda_1_loss/Mean_1Meanloss/lambda_1_loss/Mean+loss/lambda_1_loss/Mean_1/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims( 

loss/lambda_1_loss/mulMulloss/lambda_1_loss/Mean_1lambda_1_sample_weights*#
_output_shapes
:џџџџџџџџџ*
T0
b
loss/lambda_1_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss/lambda_1_loss/NotEqualNotEquallambda_1_sample_weightsloss/lambda_1_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ
y
loss/lambda_1_loss/CastCastloss/lambda_1_loss/NotEqual*#
_output_shapes
:џџџџџџџџџ*

DstT0*

SrcT0

b
loss/lambda_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss/lambda_1_loss/Mean_2Meanloss/lambda_1_loss/Castloss/lambda_1_loss/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 

loss/lambda_1_loss/truedivRealDivloss/lambda_1_loss/mulloss/lambda_1_loss/Mean_2*#
_output_shapes
:џџџџџџџџџ*
T0
d
loss/lambda_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss/lambda_1_loss/Mean_3Meanloss/lambda_1_loss/truedivloss/lambda_1_loss/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
O

loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
W
loss/mulMul
loss/mul/xloss/lambda_1_loss/Mean_3*
_output_shapes
: *
T0
`
SGD_1/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
t
SGD_1/iterations
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0	
Т
SGD_1/iterations/AssignAssignSGD_1/iterationsSGD_1/iterations/initial_value*
_output_shapes
: *
use_locking(*
T0	*#
_class
loc:@SGD_1/iterations*
validate_shape(
y
SGD_1/iterations/readIdentitySGD_1/iterations*
T0	*#
_class
loc:@SGD_1/iterations*
_output_shapes
: 
[
SGD_1/lr/initial_valueConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
l
SGD_1/lr
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Ђ
SGD_1/lr/AssignAssignSGD_1/lrSGD_1/lr/initial_value*
use_locking(*
T0*
_class
loc:@SGD_1/lr*
validate_shape(*
_output_shapes
: 
a
SGD_1/lr/readIdentitySGD_1/lr*
T0*
_class
loc:@SGD_1/lr*
_output_shapes
: 
a
SGD_1/momentum/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
r
SGD_1/momentum
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
К
SGD_1/momentum/AssignAssignSGD_1/momentumSGD_1/momentum/initial_value*
use_locking(*
T0*!
_class
loc:@SGD_1/momentum*
validate_shape(*
_output_shapes
: 
s
SGD_1/momentum/readIdentitySGD_1/momentum*
T0*!
_class
loc:@SGD_1/momentum*
_output_shapes
: 
^
SGD_1/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
o
SGD_1/decay
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Ў
SGD_1/decay/AssignAssignSGD_1/decaySGD_1/decay/initial_value*
_class
loc:@SGD_1/decay*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
j
SGD_1/decay/readIdentitySGD_1/decay*
_class
loc:@SGD_1/decay*
_output_shapes
: *
T0

lambda_1_target_1Placeholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
t
lambda_1_sample_weights_1Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
r
loss_1/lambda_1_loss/subSublambda_1/sublambda_1_target_1*'
_output_shapes
:џџџџџџџџџ*
T0
q
loss_1/lambda_1_loss/SquareSquareloss_1/lambda_1_loss/sub*
T0*'
_output_shapes
:џџџџџџџџџ
v
+loss_1/lambda_1_loss/Mean/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ж
loss_1/lambda_1_loss/MeanMeanloss_1/lambda_1_loss/Square+loss_1/lambda_1_loss/Mean/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims( 
p
-loss_1/lambda_1_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
И
loss_1/lambda_1_loss/Mean_1Meanloss_1/lambda_1_loss/Mean-loss_1/lambda_1_loss/Mean_1/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims( 

loss_1/lambda_1_loss/mulMulloss_1/lambda_1_loss/Mean_1lambda_1_sample_weights_1*
T0*#
_output_shapes
:џџџџџџџџџ
d
loss_1/lambda_1_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss_1/lambda_1_loss/NotEqualNotEquallambda_1_sample_weights_1loss_1/lambda_1_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ
}
loss_1/lambda_1_loss/CastCastloss_1/lambda_1_loss/NotEqual*

SrcT0
*#
_output_shapes
:џџџџџџџџџ*

DstT0
d
loss_1/lambda_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss_1/lambda_1_loss/Mean_2Meanloss_1/lambda_1_loss/Castloss_1/lambda_1_loss/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 

loss_1/lambda_1_loss/truedivRealDivloss_1/lambda_1_loss/mulloss_1/lambda_1_loss/Mean_2*
T0*#
_output_shapes
:џџџџџџџџџ
f
loss_1/lambda_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss_1/lambda_1_loss/Mean_3Meanloss_1/lambda_1_loss/truedivloss_1/lambda_1_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
Q
loss_1/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
]

loss_1/mulMulloss_1/mul/xloss_1/lambda_1_loss/Mean_3*
_output_shapes
: *
T0
i
y_truePlaceholder*
shape:џџџџџџџџџ*
dtype0*'
_output_shapes
:џџџџџџџџџ
g
maskPlaceholder*
shape:џџџџџџџџџ*
dtype0*'
_output_shapes
:џџџџџџџџџ
Y

loss_2/subSuby_truelambda_1/sub*'
_output_shapes
:џџџџџџџџџ*
T0
O

loss_2/AbsAbs
loss_2/sub*
T0*'
_output_shapes
:џџџџџџџџџ
R
loss_2/Less/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
`
loss_2/LessLess
loss_2/Absloss_2/Less/y*
T0*'
_output_shapes
:џџџџџџџџџ
U
loss_2/SquareSquare
loss_2/sub*'
_output_shapes
:џџџџџџџџџ*
T0
Q
loss_2/mul/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
`

loss_2/mulMulloss_2/mul/xloss_2/Square*'
_output_shapes
:џџџџџџџџџ*
T0
Q
loss_2/Abs_1Abs
loss_2/sub*
T0*'
_output_shapes
:џџџџџџџџџ
S
loss_2/sub_1/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
c
loss_2/sub_1Subloss_2/Abs_1loss_2/sub_1/y*'
_output_shapes
:џџџџџџџџџ*
T0
S
loss_2/mul_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
c
loss_2/mul_1Mulloss_2/mul_1/xloss_2/sub_1*'
_output_shapes
:џџџџџџџџџ*
T0
p
loss_2/SelectSelectloss_2/Less
loss_2/mulloss_2/mul_1*
T0*'
_output_shapes
:џџџџџџџџџ
Z
loss_2/mul_2Mulloss_2/Selectmask*'
_output_shapes
:џџџџџџџџџ*
T0
g
loss_2/Sum/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 


loss_2/SumSumloss_2/mul_2loss_2/Sum/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims( 

loss_targetPlaceholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ

lambda_1_target_2Placeholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
n
loss_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
t
lambda_1_sample_weights_2Placeholder*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ*
dtype0
j
'loss_3/loss_loss/Mean/reduction_indicesConst*
_output_shapes
: *
valueB *
dtype0

loss_3/loss_loss/MeanMean
loss_2/Sum'loss_3/loss_loss/Mean/reduction_indices*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims( *
T0
u
loss_3/loss_loss/mulMulloss_3/loss_loss/Meanloss_sample_weights*
T0*#
_output_shapes
:џџџџџџџџџ
`
loss_3/loss_loss/NotEqual/yConst*
dtype0*
_output_shapes
: *
valueB
 *    

loss_3/loss_loss/NotEqualNotEqualloss_sample_weightsloss_3/loss_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ
u
loss_3/loss_loss/CastCastloss_3/loss_loss/NotEqual*#
_output_shapes
:џџџџџџџџџ*

DstT0*

SrcT0

`
loss_3/loss_loss/ConstConst*
_output_shapes
:*
valueB: *
dtype0

loss_3/loss_loss/Mean_1Meanloss_3/loss_loss/Castloss_3/loss_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 

loss_3/loss_loss/truedivRealDivloss_3/loss_loss/mulloss_3/loss_loss/Mean_1*
T0*#
_output_shapes
:џџџџџџџџџ
b
loss_3/loss_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss_3/loss_loss/Mean_2Meanloss_3/loss_loss/truedivloss_3/loss_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
Q
loss_3/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Y

loss_3/mulMulloss_3/mul/xloss_3/loss_loss/Mean_2*
T0*
_output_shapes
: 
l
loss_3/lambda_1_loss/zeros_like	ZerosLikelambda_1/sub*'
_output_shapes
:џџџџџџџџџ*
T0
u
+loss_3/lambda_1_loss/Mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
К
loss_3/lambda_1_loss/MeanMeanloss_3/lambda_1_loss/zeros_like+loss_3/lambda_1_loss/Mean/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims( 

loss_3/lambda_1_loss/mulMulloss_3/lambda_1_loss/Meanlambda_1_sample_weights_2*
T0*#
_output_shapes
:џџџџџџџџџ
d
loss_3/lambda_1_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss_3/lambda_1_loss/NotEqualNotEquallambda_1_sample_weights_2loss_3/lambda_1_loss/NotEqual/y*#
_output_shapes
:џџџџџџџџџ*
T0
}
loss_3/lambda_1_loss/CastCastloss_3/lambda_1_loss/NotEqual*

SrcT0
*#
_output_shapes
:џџџџџџџџџ*

DstT0
d
loss_3/lambda_1_loss/ConstConst*
_output_shapes
:*
valueB: *
dtype0

loss_3/lambda_1_loss/Mean_1Meanloss_3/lambda_1_loss/Castloss_3/lambda_1_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 

loss_3/lambda_1_loss/truedivRealDivloss_3/lambda_1_loss/mulloss_3/lambda_1_loss/Mean_1*
T0*#
_output_shapes
:џџџџџџџџџ
f
loss_3/lambda_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss_3/lambda_1_loss/Mean_2Meanloss_3/lambda_1_loss/truedivloss_3/lambda_1_loss/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
S
loss_3/mul_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
loss_3/mul_1Mulloss_3/mul_1/xloss_3/lambda_1_loss/Mean_2*
_output_shapes
: *
T0
L

loss_3/addAdd
loss_3/mulloss_3/mul_1*
T0*
_output_shapes
: 
{
!metrics_2/mean_absolute_error/subSublambda_1/sublambda_1_target_2*'
_output_shapes
:џџџџџџџџџ*
T0
}
!metrics_2/mean_absolute_error/AbsAbs!metrics_2/mean_absolute_error/sub*'
_output_shapes
:џџџџџџџџџ*
T0

4metrics_2/mean_absolute_error/Mean/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ю
"metrics_2/mean_absolute_error/MeanMean!metrics_2/mean_absolute_error/Abs4metrics_2/mean_absolute_error/Mean/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:џџџџџџџџџ
m
#metrics_2/mean_absolute_error/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Г
$metrics_2/mean_absolute_error/Mean_1Mean"metrics_2/mean_absolute_error/Mean#metrics_2/mean_absolute_error/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
q
&metrics_2/mean_q/Max/reduction_indicesConst*
_output_shapes
: *
valueB :
џџџџџџџџџ*
dtype0

metrics_2/mean_q/MaxMaxlambda_1/sub&metrics_2/mean_q/Max/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims( 
`
metrics_2/mean_q/ConstConst*
valueB: *
dtype0*
_output_shapes
:

metrics_2/mean_q/MeanMeanmetrics_2/mean_q/Maxmetrics_2/mean_q/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
[
metrics_2/mean_q/Const_1Const*
valueB *
dtype0*
_output_shapes
: 

metrics_2/mean_q/Mean_1Meanmetrics_2/mean_q/Meanmetrics_2/mean_q/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 

IsVariableInitialized_29IsVariableInitializedSGD/iterations*
dtype0	*
_output_shapes
: *!
_class
loc:@SGD/iterations
y
IsVariableInitialized_30IsVariableInitializedSGD/lr*
_class
loc:@SGD/lr*
dtype0*
_output_shapes
: 

IsVariableInitialized_31IsVariableInitializedSGD/momentum*
dtype0*
_output_shapes
: *
_class
loc:@SGD/momentum

IsVariableInitialized_32IsVariableInitialized	SGD/decay*
dtype0*
_output_shapes
: *
_class
loc:@SGD/decay

IsVariableInitialized_33IsVariableInitializedSGD_1/iterations*#
_class
loc:@SGD_1/iterations*
dtype0	*
_output_shapes
: 
}
IsVariableInitialized_34IsVariableInitializedSGD_1/lr*
_class
loc:@SGD_1/lr*
dtype0*
_output_shapes
: 

IsVariableInitialized_35IsVariableInitializedSGD_1/momentum*
_output_shapes
: *!
_class
loc:@SGD_1/momentum*
dtype0

IsVariableInitialized_36IsVariableInitializedSGD_1/decay*
_class
loc:@SGD_1/decay*
dtype0*
_output_shapes
: 
И
init_1NoOp^SGD/iterations/Assign^SGD/lr/Assign^SGD/momentum/Assign^SGD/decay/Assign^SGD_1/iterations/Assign^SGD_1/lr/Assign^SGD_1/momentum/Assign^SGD_1/decay/Assign"ЮaГ1%     ­§ц	ч2ЄnзAJЄЪ
њ
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
ы
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype
is_initialized
"
dtypetype
:
Less
x"T
y"T
z
"
Ttype:
2	
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	

M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
D
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
1
Square
x"T
y"T"
Ttype:

2	
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*1.5.02v1.5.0-0-g37aa430d84ў

permute_1_inputPlaceholder*
dtype0*/
_output_shapes
:џџџџџџџџџ(P*$
shape:џџџџџџџџџ(P
q
permute_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:

permute_1/transpose	Transposepermute_1_inputpermute_1/transpose/perm*
Tperm0*
T0*/
_output_shapes
:џџџџџџџџџ(P
v
conv2d_1/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
`
conv2d_1/random_uniform/minConst*
valueB
 *ѓЕН*
dtype0*
_output_shapes
: 
`
conv2d_1/random_uniform/maxConst*
valueB
 *ѓЕ=*
dtype0*
_output_shapes
: 
В
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
seedБџх)*
T0*
dtype0*&
_output_shapes
:*
seed2ўЏ
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
_output_shapes
: *
T0

conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*
T0*&
_output_shapes
:

conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*&
_output_shapes
:*
T0

conv2d_1/kernel
VariableV2*
shared_name *
dtype0*&
_output_shapes
:*
	container *
shape:
Ш
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:

conv2d_1/kernel/readIdentityconv2d_1/kernel*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:
[
conv2d_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
conv2d_1/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
­
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
t
conv2d_1/bias/readIdentityconv2d_1/bias*
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:
s
"conv2d_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ђ
conv2d_1/convolutionConv2Dpermute_1/transposeconv2d_1/kernel/read*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ	*
	dilations
*
T0*
strides
*
data_formatNHWC

conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*/
_output_shapes
:џџџџџџџџџ	*
T0*
data_formatNHWC
e
activation_1/ReluReluconv2d_1/BiasAdd*/
_output_shapes
:џџџџџџџџџ	*
T0
v
conv2d_2/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
`
conv2d_2/random_uniform/minConst*
valueB
 *   О*
dtype0*
_output_shapes
: 
`
conv2d_2/random_uniform/maxConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
В
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
seedБџх)*
T0*
dtype0*&
_output_shapes
:*
seed2эњ
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
_output_shapes
: *
T0

conv2d_2/random_uniform/mulMul%conv2d_2/random_uniform/RandomUniformconv2d_2/random_uniform/sub*
T0*&
_output_shapes
:

conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*&
_output_shapes
:*
T0

conv2d_2/kernel
VariableV2*
shape:*
shared_name *
dtype0*&
_output_shapes
:*
	container 
Ш
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0

conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
[
conv2d_2/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
conv2d_2/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
­
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
t
conv2d_2/bias/readIdentityconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
_output_shapes
:*
T0
s
"conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
№
conv2d_2/convolutionConv2Dactivation_1/Reluconv2d_2/kernel/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ*
	dilations


conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*/
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
e
activation_2/ReluReluconv2d_2/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ
v
conv2d_3/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
`
conv2d_3/random_uniform/minConst*
valueB
 *:ЭО*
dtype0*
_output_shapes
: 
`
conv2d_3/random_uniform/maxConst*
valueB
 *:Э>*
dtype0*
_output_shapes
: 
В
%conv2d_3/random_uniform/RandomUniformRandomUniformconv2d_3/random_uniform/shape*
T0*
dtype0*&
_output_shapes
:*
seed2ўк*
seedБџх)
}
conv2d_3/random_uniform/subSubconv2d_3/random_uniform/maxconv2d_3/random_uniform/min*
T0*
_output_shapes
: 

conv2d_3/random_uniform/mulMul%conv2d_3/random_uniform/RandomUniformconv2d_3/random_uniform/sub*&
_output_shapes
:*
T0

conv2d_3/random_uniformAddconv2d_3/random_uniform/mulconv2d_3/random_uniform/min*
T0*&
_output_shapes
:

conv2d_3/kernel
VariableV2*
shape:*
shared_name *
dtype0*&
_output_shapes
:*
	container 
Ш
conv2d_3/kernel/AssignAssignconv2d_3/kernelconv2d_3/random_uniform*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel

conv2d_3/kernel/readIdentityconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:*
T0
[
conv2d_3/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
conv2d_3/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
­
conv2d_3/bias/AssignAssignconv2d_3/biasconv2d_3/Const*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv2d_3/bias*
validate_shape(
t
conv2d_3/bias/readIdentityconv2d_3/bias*
T0* 
_class
loc:@conv2d_3/bias*
_output_shapes
:
s
"conv2d_3/convolution/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
№
conv2d_3/convolutionConv2Dactivation_2/Reluconv2d_3/kernel/read*/
_output_shapes
:џџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID

conv2d_3/BiasAddBiasAddconv2d_3/convolutionconv2d_3/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ
e
activation_3/ReluReluconv2d_3/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ
`
flatten_1/ShapeShapeactivation_3/Relu*
T0*
out_type0*
_output_shapes
:
g
flatten_1/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
i
flatten_1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
i
flatten_1/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Џ
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
Y
flatten_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
~
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
\
flatten_1/stack/0Const*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*
_output_shapes
:*
T0*

axis *
N

flatten_1/ReshapeReshapeactivation_3/Reluflatten_1/stack*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
m
dense_1/random_uniform/shapeConst*
valueB"`      *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *b'О*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *b'>*
dtype0*
_output_shapes
: 
Ј
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
seedБџх)*
T0*
dtype0*
_output_shapes
:	`*
seed2мпK
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 

dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes
:	`

dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes
:	`

dense_1/kernel
VariableV2*
dtype0*
_output_shapes
:	`*
	container *
shape:	`*
shared_name 
Н
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes
:	`
|
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	`
\
dense_1/ConstConst*
valueB*    *
dtype0*
_output_shapes	
:
z
dense_1/bias
VariableV2*
_output_shapes	
:*
	container *
shape:*
shared_name *
dtype0
Њ
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:
r
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:

dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
]
activation_4/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
m
dense_2/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
_
dense_2/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ЃЎXО
_
dense_2/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ЃЎX>
Љ
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
T0*
dtype0*
_output_shapes
:	*
seed2цэ*
seedБџх)
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 

dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0*
_output_shapes
:	

dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes
:	

dense_2/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes
:	*
	container *
shape:	
Н
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes
:	
|
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
:	
Z
dense_2/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_2/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Љ
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@dense_2/bias
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:

dense_2/MatMulMatMulactivation_4/Reludense_2/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
d
activation_5/IdentityIdentitydense_2/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
m
dense_3/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_3/random_uniform/minConst*
valueB
 *ђъ-П*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
valueB
 *ђъ-?*
dtype0*
_output_shapes
: 
Ї
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
_output_shapes

:*
seed2јh*
seedБџх)*
T0*
dtype0
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 

dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
_output_shapes

:*
T0
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
_output_shapes

:*
T0

dense_3/kernel
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
М
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:
{
dense_3/kernel/readIdentitydense_3/kernel*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:
Z
dense_3/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_3/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Љ
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
q
dense_3/bias/readIdentitydense_3/bias*
T0*
_class
loc:@dense_3/bias*
_output_shapes
:

dense_3/MatMulMatMuldense_2/BiasAdddense_3/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
m
lambda_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
o
lambda_1/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
o
lambda_1/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Д
lambda_1/strided_sliceStridedSlicedense_3/BiasAddlambda_1/strided_slice/stacklambda_1/strided_slice/stack_1lambda_1/strided_slice/stack_2*
end_mask*#
_output_shapes
:џџџџџџџџџ*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
b
lambda_1/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

lambda_1/ExpandDims
ExpandDimslambda_1/strided_slicelambda_1/ExpandDims/dim*'
_output_shapes
:џџџџџџџџџ*

Tdim0*
T0
o
lambda_1/strided_slice_1/stackConst*
valueB"       *
dtype0*
_output_shapes
:
q
 lambda_1/strided_slice_1/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
q
 lambda_1/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Р
lambda_1/strided_slice_1StridedSlicedense_3/BiasAddlambda_1/strided_slice_1/stack lambda_1/strided_slice_1/stack_1 lambda_1/strided_slice_1/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:џџџџџџџџџ*
T0*
Index0
t
lambda_1/addAddlambda_1/ExpandDimslambda_1/strided_slice_1*
T0*'
_output_shapes
:џџџџџџџџџ
o
lambda_1/strided_slice_2/stackConst*
valueB"       *
dtype0*
_output_shapes
:
q
 lambda_1/strided_slice_2/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
q
 lambda_1/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
Р
lambda_1/strided_slice_2StridedSlicedense_3/BiasAddlambda_1/strided_slice_2/stack lambda_1/strided_slice_2/stack_1 lambda_1/strided_slice_2/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:џџџџџџџџџ*
T0*
Index0
_
lambda_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:

lambda_1/MeanMeanlambda_1/strided_slice_2lambda_1/Const*
_output_shapes

:*

Tidx0*
	keep_dims(*
T0
b
lambda_1/subSublambda_1/addlambda_1/Mean*'
_output_shapes
:џџџџџџџџџ*
T0
_
Adam/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
s
Adam/iterations
VariableV2*
shared_name *
dtype0	*
_output_shapes
: *
	container *
shape: 
О
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*"
_class
loc:@Adam/iterations*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	
v
Adam/iterations/readIdentityAdam/iterations*"
_class
loc:@Adam/iterations*
_output_shapes
: *
T0	
Z
Adam/lr/initial_valueConst*
valueB
 *o9*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 

Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
use_locking(*
T0*
_class
loc:@Adam/lr*
validate_shape(*
_output_shapes
: 
^
Adam/lr/readIdentityAdam/lr*
T0*
_class
loc:@Adam/lr*
_output_shapes
: 
^
Adam/beta_1/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
o
Adam/beta_1
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
Ў
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
use_locking(*
T0*
_class
loc:@Adam/beta_1*
validate_shape(*
_output_shapes
: 
j
Adam/beta_1/readIdentityAdam/beta_1*
T0*
_class
loc:@Adam/beta_1*
_output_shapes
: 
^
Adam/beta_2/initial_valueConst*
valueB
 *wО?*
dtype0*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
Ў
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/beta_2
j
Adam/beta_2/readIdentityAdam/beta_2*
_output_shapes
: *
T0*
_class
loc:@Adam/beta_2
]
Adam/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n

Adam/decay
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
Њ
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
use_locking(*
T0*
_class
loc:@Adam/decay*
validate_shape(*
_output_shapes
: 
g
Adam/decay/readIdentity
Adam/decay*
_output_shapes
: *
T0*
_class
loc:@Adam/decay

permute_1_input_1Placeholder*
dtype0*/
_output_shapes
:џџџџџџџџџ(P*$
shape:џџџџџџџџџ(P
s
permute_1_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:

permute_1_1/transpose	Transposepermute_1_input_1permute_1_1/transpose/perm*/
_output_shapes
:џџџџџџџџџ(P*
Tperm0*
T0
x
conv2d_1_1/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
b
conv2d_1_1/random_uniform/minConst*
valueB
 *ѓЕН*
dtype0*
_output_shapes
: 
b
conv2d_1_1/random_uniform/maxConst*
valueB
 *ѓЕ=*
dtype0*
_output_shapes
: 
Ж
'conv2d_1_1/random_uniform/RandomUniformRandomUniformconv2d_1_1/random_uniform/shape*
seedБџх)*
T0*
dtype0*&
_output_shapes
:*
seed2гММ

conv2d_1_1/random_uniform/subSubconv2d_1_1/random_uniform/maxconv2d_1_1/random_uniform/min*
_output_shapes
: *
T0

conv2d_1_1/random_uniform/mulMul'conv2d_1_1/random_uniform/RandomUniformconv2d_1_1/random_uniform/sub*
T0*&
_output_shapes
:

conv2d_1_1/random_uniformAddconv2d_1_1/random_uniform/mulconv2d_1_1/random_uniform/min*
T0*&
_output_shapes
:

conv2d_1_1/kernel
VariableV2*
dtype0*&
_output_shapes
:*
	container *
shape:*
shared_name 
а
conv2d_1_1/kernel/AssignAssignconv2d_1_1/kernelconv2d_1_1/random_uniform*
use_locking(*
T0*$
_class
loc:@conv2d_1_1/kernel*
validate_shape(*&
_output_shapes
:

conv2d_1_1/kernel/readIdentityconv2d_1_1/kernel*
T0*$
_class
loc:@conv2d_1_1/kernel*&
_output_shapes
:
]
conv2d_1_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
{
conv2d_1_1/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Е
conv2d_1_1/bias/AssignAssignconv2d_1_1/biasconv2d_1_1/Const*
T0*"
_class
loc:@conv2d_1_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
z
conv2d_1_1/bias/readIdentityconv2d_1_1/bias*
T0*"
_class
loc:@conv2d_1_1/bias*
_output_shapes
:
u
$conv2d_1_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ј
conv2d_1_1/convolutionConv2Dpermute_1_1/transposeconv2d_1_1/kernel/read*
paddingVALID*/
_output_shapes
:џџџџџџџџџ	*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

conv2d_1_1/BiasAddBiasAddconv2d_1_1/convolutionconv2d_1_1/bias/read*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ	*
T0
i
activation_1_1/ReluReluconv2d_1_1/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ	
x
conv2d_2_1/random_uniform/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0
b
conv2d_2_1/random_uniform/minConst*
valueB
 *   О*
dtype0*
_output_shapes
: 
b
conv2d_2_1/random_uniform/maxConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
Ж
'conv2d_2_1/random_uniform/RandomUniformRandomUniformconv2d_2_1/random_uniform/shape*
T0*
dtype0*&
_output_shapes
:*
seed2ћшТ*
seedБџх)

conv2d_2_1/random_uniform/subSubconv2d_2_1/random_uniform/maxconv2d_2_1/random_uniform/min*
T0*
_output_shapes
: 

conv2d_2_1/random_uniform/mulMul'conv2d_2_1/random_uniform/RandomUniformconv2d_2_1/random_uniform/sub*
T0*&
_output_shapes
:

conv2d_2_1/random_uniformAddconv2d_2_1/random_uniform/mulconv2d_2_1/random_uniform/min*&
_output_shapes
:*
T0

conv2d_2_1/kernel
VariableV2*
dtype0*&
_output_shapes
:*
	container *
shape:*
shared_name 
а
conv2d_2_1/kernel/AssignAssignconv2d_2_1/kernelconv2d_2_1/random_uniform*&
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@conv2d_2_1/kernel*
validate_shape(

conv2d_2_1/kernel/readIdentityconv2d_2_1/kernel*&
_output_shapes
:*
T0*$
_class
loc:@conv2d_2_1/kernel
]
conv2d_2_1/ConstConst*
dtype0*
_output_shapes
:*
valueB*    
{
conv2d_2_1/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Е
conv2d_2_1/bias/AssignAssignconv2d_2_1/biasconv2d_2_1/Const*
use_locking(*
T0*"
_class
loc:@conv2d_2_1/bias*
validate_shape(*
_output_shapes
:
z
conv2d_2_1/bias/readIdentityconv2d_2_1/bias*
T0*"
_class
loc:@conv2d_2_1/bias*
_output_shapes
:
u
$conv2d_2_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
і
conv2d_2_1/convolutionConv2Dactivation_1_1/Reluconv2d_2_1/kernel/read*/
_output_shapes
:џџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID

conv2d_2_1/BiasAddBiasAddconv2d_2_1/convolutionconv2d_2_1/bias/read*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ*
T0
i
activation_2_1/ReluReluconv2d_2_1/BiasAdd*/
_output_shapes
:џџџџџџџџџ*
T0
x
conv2d_3_1/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
b
conv2d_3_1/random_uniform/minConst*
_output_shapes
: *
valueB
 *:ЭО*
dtype0
b
conv2d_3_1/random_uniform/maxConst*
valueB
 *:Э>*
dtype0*
_output_shapes
: 
Е
'conv2d_3_1/random_uniform/RandomUniformRandomUniformconv2d_3_1/random_uniform/shape*
dtype0*&
_output_shapes
:*
seed2жЙM*
seedБџх)*
T0

conv2d_3_1/random_uniform/subSubconv2d_3_1/random_uniform/maxconv2d_3_1/random_uniform/min*
T0*
_output_shapes
: 

conv2d_3_1/random_uniform/mulMul'conv2d_3_1/random_uniform/RandomUniformconv2d_3_1/random_uniform/sub*&
_output_shapes
:*
T0

conv2d_3_1/random_uniformAddconv2d_3_1/random_uniform/mulconv2d_3_1/random_uniform/min*&
_output_shapes
:*
T0

conv2d_3_1/kernel
VariableV2*
dtype0*&
_output_shapes
:*
	container *
shape:*
shared_name 
а
conv2d_3_1/kernel/AssignAssignconv2d_3_1/kernelconv2d_3_1/random_uniform*
use_locking(*
T0*$
_class
loc:@conv2d_3_1/kernel*
validate_shape(*&
_output_shapes
:

conv2d_3_1/kernel/readIdentityconv2d_3_1/kernel*
T0*$
_class
loc:@conv2d_3_1/kernel*&
_output_shapes
:
]
conv2d_3_1/ConstConst*
_output_shapes
:*
valueB*    *
dtype0
{
conv2d_3_1/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Е
conv2d_3_1/bias/AssignAssignconv2d_3_1/biasconv2d_3_1/Const*
use_locking(*
T0*"
_class
loc:@conv2d_3_1/bias*
validate_shape(*
_output_shapes
:
z
conv2d_3_1/bias/readIdentityconv2d_3_1/bias*
T0*"
_class
loc:@conv2d_3_1/bias*
_output_shapes
:
u
$conv2d_3_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
і
conv2d_3_1/convolutionConv2Dactivation_2_1/Reluconv2d_3_1/kernel/read*
paddingVALID*/
_output_shapes
:џџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

conv2d_3_1/BiasAddBiasAddconv2d_3_1/convolutionconv2d_3_1/bias/read*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ*
T0
i
activation_3_1/ReluReluconv2d_3_1/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ
d
flatten_1_1/ShapeShapeactivation_3_1/Relu*
_output_shapes
:*
T0*
out_type0
i
flatten_1_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
k
!flatten_1_1/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
k
!flatten_1_1/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Й
flatten_1_1/strided_sliceStridedSliceflatten_1_1/Shapeflatten_1_1/strided_slice/stack!flatten_1_1/strided_slice/stack_1!flatten_1_1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
[
flatten_1_1/ConstConst*
dtype0*
_output_shapes
:*
valueB: 

flatten_1_1/ProdProdflatten_1_1/strided_sliceflatten_1_1/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
^
flatten_1_1/stack/0Const*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
z
flatten_1_1/stackPackflatten_1_1/stack/0flatten_1_1/Prod*
T0*

axis *
N*
_output_shapes
:

flatten_1_1/ReshapeReshapeactivation_3_1/Reluflatten_1_1/stack*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
o
dense_1_1/random_uniform/shapeConst*
valueB"`      *
dtype0*
_output_shapes
:
a
dense_1_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *b'О
a
dense_1_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *b'>
­
&dense_1_1/random_uniform/RandomUniformRandomUniformdense_1_1/random_uniform/shape*
seedБџх)*
T0*
dtype0*
_output_shapes
:	`*
seed2с­Щ

dense_1_1/random_uniform/subSubdense_1_1/random_uniform/maxdense_1_1/random_uniform/min*
T0*
_output_shapes
: 

dense_1_1/random_uniform/mulMul&dense_1_1/random_uniform/RandomUniformdense_1_1/random_uniform/sub*
T0*
_output_shapes
:	`

dense_1_1/random_uniformAdddense_1_1/random_uniform/muldense_1_1/random_uniform/min*
T0*
_output_shapes
:	`

dense_1_1/kernel
VariableV2*
shape:	`*
shared_name *
dtype0*
_output_shapes
:	`*
	container 
Х
dense_1_1/kernel/AssignAssigndense_1_1/kerneldense_1_1/random_uniform*#
_class
loc:@dense_1_1/kernel*
validate_shape(*
_output_shapes
:	`*
use_locking(*
T0

dense_1_1/kernel/readIdentitydense_1_1/kernel*
T0*#
_class
loc:@dense_1_1/kernel*
_output_shapes
:	`
^
dense_1_1/ConstConst*
_output_shapes	
:*
valueB*    *
dtype0
|
dense_1_1/bias
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
В
dense_1_1/bias/AssignAssigndense_1_1/biasdense_1_1/Const*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@dense_1_1/bias
x
dense_1_1/bias/readIdentitydense_1_1/bias*
_output_shapes	
:*
T0*!
_class
loc:@dense_1_1/bias

dense_1_1/MatMulMatMulflatten_1_1/Reshapedense_1_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 

dense_1_1/BiasAddBiasAdddense_1_1/MatMuldense_1_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
a
activation_4_1/ReluReludense_1_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
o
dense_2_1/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
a
dense_2_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ЃЎXО
a
dense_2_1/random_uniform/maxConst*
valueB
 *ЃЎX>*
dtype0*
_output_shapes
: 
­
&dense_2_1/random_uniform/RandomUniformRandomUniformdense_2_1/random_uniform/shape*
T0*
dtype0*
_output_shapes
:	*
seed2рФ*
seedБџх)

dense_2_1/random_uniform/subSubdense_2_1/random_uniform/maxdense_2_1/random_uniform/min*
_output_shapes
: *
T0

dense_2_1/random_uniform/mulMul&dense_2_1/random_uniform/RandomUniformdense_2_1/random_uniform/sub*
T0*
_output_shapes
:	

dense_2_1/random_uniformAdddense_2_1/random_uniform/muldense_2_1/random_uniform/min*
T0*
_output_shapes
:	

dense_2_1/kernel
VariableV2*
shape:	*
shared_name *
dtype0*
_output_shapes
:	*
	container 
Х
dense_2_1/kernel/AssignAssigndense_2_1/kerneldense_2_1/random_uniform*
T0*#
_class
loc:@dense_2_1/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(

dense_2_1/kernel/readIdentitydense_2_1/kernel*
T0*#
_class
loc:@dense_2_1/kernel*
_output_shapes
:	
\
dense_2_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
z
dense_2_1/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Б
dense_2_1/bias/AssignAssigndense_2_1/biasdense_2_1/Const*
T0*!
_class
loc:@dense_2_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
w
dense_2_1/bias/readIdentitydense_2_1/bias*
T0*!
_class
loc:@dense_2_1/bias*
_output_shapes
:

dense_2_1/MatMulMatMulactivation_4_1/Reludense_2_1/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

dense_2_1/BiasAddBiasAdddense_2_1/MatMuldense_2_1/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
o
dense_3_1/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
a
dense_3_1/random_uniform/minConst*
valueB
 *ђъ-П*
dtype0*
_output_shapes
: 
a
dense_3_1/random_uniform/maxConst*
valueB
 *ђъ-?*
dtype0*
_output_shapes
: 
Ќ
&dense_3_1/random_uniform/RandomUniformRandomUniformdense_3_1/random_uniform/shape*
T0*
dtype0*
_output_shapes

:*
seed2ёА*
seedБџх)

dense_3_1/random_uniform/subSubdense_3_1/random_uniform/maxdense_3_1/random_uniform/min*
T0*
_output_shapes
: 

dense_3_1/random_uniform/mulMul&dense_3_1/random_uniform/RandomUniformdense_3_1/random_uniform/sub*
T0*
_output_shapes

:

dense_3_1/random_uniformAdddense_3_1/random_uniform/muldense_3_1/random_uniform/min*
_output_shapes

:*
T0

dense_3_1/kernel
VariableV2*
shape
:*
shared_name *
dtype0*
_output_shapes

:*
	container 
Ф
dense_3_1/kernel/AssignAssigndense_3_1/kerneldense_3_1/random_uniform*
T0*#
_class
loc:@dense_3_1/kernel*
validate_shape(*
_output_shapes

:*
use_locking(

dense_3_1/kernel/readIdentitydense_3_1/kernel*
T0*#
_class
loc:@dense_3_1/kernel*
_output_shapes

:
\
dense_3_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
z
dense_3_1/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Б
dense_3_1/bias/AssignAssigndense_3_1/biasdense_3_1/Const*
use_locking(*
T0*!
_class
loc:@dense_3_1/bias*
validate_shape(*
_output_shapes
:
w
dense_3_1/bias/readIdentitydense_3_1/bias*
T0*!
_class
loc:@dense_3_1/bias*
_output_shapes
:

dense_3_1/MatMulMatMuldense_2_1/BiasAdddense_3_1/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

dense_3_1/BiasAddBiasAdddense_3_1/MatMuldense_3_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
o
lambda_1_1/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
q
 lambda_1_1/strided_slice/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
q
 lambda_1_1/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
О
lambda_1_1/strided_sliceStridedSlicedense_3_1/BiasAddlambda_1_1/strided_slice/stack lambda_1_1/strided_slice/stack_1 lambda_1_1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*#
_output_shapes
:џџџџџџџџџ
d
lambda_1_1/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

lambda_1_1/ExpandDims
ExpandDimslambda_1_1/strided_slicelambda_1_1/ExpandDims/dim*'
_output_shapes
:џџџџџџџџџ*

Tdim0*
T0
q
 lambda_1_1/strided_slice_1/stackConst*
_output_shapes
:*
valueB"       *
dtype0
s
"lambda_1_1/strided_slice_1/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
s
"lambda_1_1/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Ъ
lambda_1_1/strided_slice_1StridedSlicedense_3_1/BiasAdd lambda_1_1/strided_slice_1/stack"lambda_1_1/strided_slice_1/stack_1"lambda_1_1/strided_slice_1/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*'
_output_shapes
:џџџџџџџџџ*
Index0*
T0
z
lambda_1_1/addAddlambda_1_1/ExpandDimslambda_1_1/strided_slice_1*'
_output_shapes
:џџџџџџџџџ*
T0
q
 lambda_1_1/strided_slice_2/stackConst*
valueB"       *
dtype0*
_output_shapes
:
s
"lambda_1_1/strided_slice_2/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0
s
"lambda_1_1/strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Ъ
lambda_1_1/strided_slice_2StridedSlicedense_3_1/BiasAdd lambda_1_1/strided_slice_2/stack"lambda_1_1/strided_slice_2/stack_1"lambda_1_1/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:џџџџџџџџџ
a
lambda_1_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:

lambda_1_1/MeanMeanlambda_1_1/strided_slice_2lambda_1_1/Const*

Tidx0*
	keep_dims(*
T0*
_output_shapes

:
h
lambda_1_1/subSublambda_1_1/addlambda_1_1/Mean*'
_output_shapes
:џџџџџџџџџ*
T0

IsVariableInitializedIsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_1IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_2IsVariableInitializedconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_3IsVariableInitializedconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_4IsVariableInitializedconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_5IsVariableInitializedconv2d_3/bias*
_output_shapes
: * 
_class
loc:@conv2d_3/bias*
dtype0

IsVariableInitialized_6IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_7IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializeddense_2/kernel*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_9IsVariableInitializeddense_2/bias*
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_10IsVariableInitializeddense_3/kernel*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_11IsVariableInitializeddense_3/bias*
_output_shapes
: *
_class
loc:@dense_3/bias*
dtype0

IsVariableInitialized_12IsVariableInitializedAdam/iterations*"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
{
IsVariableInitialized_13IsVariableInitializedAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 

IsVariableInitialized_14IsVariableInitializedAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 

IsVariableInitialized_15IsVariableInitializedAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_16IsVariableInitialized
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 

IsVariableInitialized_17IsVariableInitializedconv2d_1_1/kernel*$
_class
loc:@conv2d_1_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_18IsVariableInitializedconv2d_1_1/bias*"
_class
loc:@conv2d_1_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_19IsVariableInitializedconv2d_2_1/kernel*
dtype0*
_output_shapes
: *$
_class
loc:@conv2d_2_1/kernel

IsVariableInitialized_20IsVariableInitializedconv2d_2_1/bias*"
_class
loc:@conv2d_2_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_21IsVariableInitializedconv2d_3_1/kernel*$
_class
loc:@conv2d_3_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_22IsVariableInitializedconv2d_3_1/bias*"
_class
loc:@conv2d_3_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_23IsVariableInitializeddense_1_1/kernel*#
_class
loc:@dense_1_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_24IsVariableInitializeddense_1_1/bias*!
_class
loc:@dense_1_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_25IsVariableInitializeddense_2_1/kernel*
dtype0*
_output_shapes
: *#
_class
loc:@dense_2_1/kernel

IsVariableInitialized_26IsVariableInitializeddense_2_1/bias*!
_class
loc:@dense_2_1/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_27IsVariableInitializeddense_3_1/kernel*#
_class
loc:@dense_3_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_28IsVariableInitializeddense_3_1/bias*!
_class
loc:@dense_3_1/bias*
dtype0*
_output_shapes
: 
Р
initNoOp^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^conv2d_2/kernel/Assign^conv2d_2/bias/Assign^conv2d_3/kernel/Assign^conv2d_3/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^Adam/iterations/Assign^Adam/lr/Assign^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^conv2d_1_1/kernel/Assign^conv2d_1_1/bias/Assign^conv2d_2_1/kernel/Assign^conv2d_2_1/bias/Assign^conv2d_3_1/kernel/Assign^conv2d_3_1/bias/Assign^dense_1_1/kernel/Assign^dense_1_1/bias/Assign^dense_2_1/kernel/Assign^dense_2_1/bias/Assign^dense_3_1/kernel/Assign^dense_3_1/bias/Assign
l
PlaceholderPlaceholder*
shape:*
dtype0*&
_output_shapes
:
А
AssignAssignconv2d_1_1/kernelPlaceholder*
use_locking( *
T0*$
_class
loc:@conv2d_1_1/kernel*
validate_shape(*&
_output_shapes
:
V
Placeholder_1Placeholder*
dtype0*
_output_shapes
:*
shape:
Є
Assign_1Assignconv2d_1_1/biasPlaceholder_1*
use_locking( *
T0*"
_class
loc:@conv2d_1_1/bias*
validate_shape(*
_output_shapes
:
n
Placeholder_2Placeholder*
dtype0*&
_output_shapes
:*
shape:
Д
Assign_2Assignconv2d_2_1/kernelPlaceholder_2*
validate_shape(*&
_output_shapes
:*
use_locking( *
T0*$
_class
loc:@conv2d_2_1/kernel
V
Placeholder_3Placeholder*
_output_shapes
:*
shape:*
dtype0
Є
Assign_3Assignconv2d_2_1/biasPlaceholder_3*
use_locking( *
T0*"
_class
loc:@conv2d_2_1/bias*
validate_shape(*
_output_shapes
:
n
Placeholder_4Placeholder*
dtype0*&
_output_shapes
:*
shape:
Д
Assign_4Assignconv2d_3_1/kernelPlaceholder_4*
T0*$
_class
loc:@conv2d_3_1/kernel*
validate_shape(*&
_output_shapes
:*
use_locking( 
V
Placeholder_5Placeholder*
dtype0*
_output_shapes
:*
shape:
Є
Assign_5Assignconv2d_3_1/biasPlaceholder_5*
use_locking( *
T0*"
_class
loc:@conv2d_3_1/bias*
validate_shape(*
_output_shapes
:
`
Placeholder_6Placeholder*
dtype0*
_output_shapes
:	`*
shape:	`
Ћ
Assign_6Assigndense_1_1/kernelPlaceholder_6*
validate_shape(*
_output_shapes
:	`*
use_locking( *
T0*#
_class
loc:@dense_1_1/kernel
X
Placeholder_7Placeholder*
shape:*
dtype0*
_output_shapes	
:
Ѓ
Assign_7Assigndense_1_1/biasPlaceholder_7*
_output_shapes	
:*
use_locking( *
T0*!
_class
loc:@dense_1_1/bias*
validate_shape(
`
Placeholder_8Placeholder*
dtype0*
_output_shapes
:	*
shape:	
Ћ
Assign_8Assigndense_2_1/kernelPlaceholder_8*
use_locking( *
T0*#
_class
loc:@dense_2_1/kernel*
validate_shape(*
_output_shapes
:	
V
Placeholder_9Placeholder*
shape:*
dtype0*
_output_shapes
:
Ђ
Assign_9Assigndense_2_1/biasPlaceholder_9*
use_locking( *
T0*!
_class
loc:@dense_2_1/bias*
validate_shape(*
_output_shapes
:
_
Placeholder_10Placeholder*
dtype0*
_output_shapes

:*
shape
:
Ќ
	Assign_10Assigndense_3_1/kernelPlaceholder_10*
use_locking( *
T0*#
_class
loc:@dense_3_1/kernel*
validate_shape(*
_output_shapes

:
W
Placeholder_11Placeholder*
shape:*
dtype0*
_output_shapes
:
Є
	Assign_11Assigndense_3_1/biasPlaceholder_11*
use_locking( *
T0*!
_class
loc:@dense_3_1/bias*
validate_shape(*
_output_shapes
:
^
SGD/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
r
SGD/iterations
VariableV2*
dtype0	*
_output_shapes
: *
	container *
shape: *
shared_name 
К
SGD/iterations/AssignAssignSGD/iterationsSGD/iterations/initial_value*!
_class
loc:@SGD/iterations*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	
s
SGD/iterations/readIdentitySGD/iterations*
T0	*!
_class
loc:@SGD/iterations*
_output_shapes
: 
Y
SGD/lr/initial_valueConst*
_output_shapes
: *
valueB
 *
з#<*
dtype0
j
SGD/lr
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 

SGD/lr/AssignAssignSGD/lrSGD/lr/initial_value*
use_locking(*
T0*
_class
loc:@SGD/lr*
validate_shape(*
_output_shapes
: 
[
SGD/lr/readIdentitySGD/lr*
_output_shapes
: *
T0*
_class
loc:@SGD/lr
_
SGD/momentum/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
p
SGD/momentum
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
В
SGD/momentum/AssignAssignSGD/momentumSGD/momentum/initial_value*
use_locking(*
T0*
_class
loc:@SGD/momentum*
validate_shape(*
_output_shapes
: 
m
SGD/momentum/readIdentitySGD/momentum*
T0*
_class
loc:@SGD/momentum*
_output_shapes
: 
\
SGD/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
m
	SGD/decay
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
І
SGD/decay/AssignAssign	SGD/decaySGD/decay/initial_value*
use_locking(*
T0*
_class
loc:@SGD/decay*
validate_shape(*
_output_shapes
: 
d
SGD/decay/readIdentity	SGD/decay*
T0*
_class
loc:@SGD/decay*
_output_shapes
: 

lambda_1_targetPlaceholder*%
shape:џџџџџџџџџџџџџџџџџџ*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
r
lambda_1_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
p
loss/lambda_1_loss/subSublambda_1_1/sublambda_1_target*
T0*'
_output_shapes
:џџџџџџџџџ
m
loss/lambda_1_loss/SquareSquareloss/lambda_1_loss/sub*
T0*'
_output_shapes
:џџџџџџџџџ
t
)loss/lambda_1_loss/Mean/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
А
loss/lambda_1_loss/MeanMeanloss/lambda_1_loss/Square)loss/lambda_1_loss/Mean/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims( 
n
+loss/lambda_1_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
В
loss/lambda_1_loss/Mean_1Meanloss/lambda_1_loss/Mean+loss/lambda_1_loss/Mean_1/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:џџџџџџџџџ

loss/lambda_1_loss/mulMulloss/lambda_1_loss/Mean_1lambda_1_sample_weights*
T0*#
_output_shapes
:џџџџџџџџџ
b
loss/lambda_1_loss/NotEqual/yConst*
dtype0*
_output_shapes
: *
valueB
 *    

loss/lambda_1_loss/NotEqualNotEquallambda_1_sample_weightsloss/lambda_1_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ
y
loss/lambda_1_loss/CastCastloss/lambda_1_loss/NotEqual*#
_output_shapes
:џџџџџџџџџ*

DstT0*

SrcT0

b
loss/lambda_1_loss/ConstConst*
dtype0*
_output_shapes
:*
valueB: 

loss/lambda_1_loss/Mean_2Meanloss/lambda_1_loss/Castloss/lambda_1_loss/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0

loss/lambda_1_loss/truedivRealDivloss/lambda_1_loss/mulloss/lambda_1_loss/Mean_2*
T0*#
_output_shapes
:џџџџџџџџџ
d
loss/lambda_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss/lambda_1_loss/Mean_3Meanloss/lambda_1_loss/truedivloss/lambda_1_loss/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
W
loss/mulMul
loss/mul/xloss/lambda_1_loss/Mean_3*
T0*
_output_shapes
: 
`
SGD_1/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
t
SGD_1/iterations
VariableV2*
shared_name *
dtype0	*
_output_shapes
: *
	container *
shape: 
Т
SGD_1/iterations/AssignAssignSGD_1/iterationsSGD_1/iterations/initial_value*
use_locking(*
T0	*#
_class
loc:@SGD_1/iterations*
validate_shape(*
_output_shapes
: 
y
SGD_1/iterations/readIdentitySGD_1/iterations*
_output_shapes
: *
T0	*#
_class
loc:@SGD_1/iterations
[
SGD_1/lr/initial_valueConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
l
SGD_1/lr
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Ђ
SGD_1/lr/AssignAssignSGD_1/lrSGD_1/lr/initial_value*
_class
loc:@SGD_1/lr*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
a
SGD_1/lr/readIdentitySGD_1/lr*
T0*
_class
loc:@SGD_1/lr*
_output_shapes
: 
a
SGD_1/momentum/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
r
SGD_1/momentum
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
К
SGD_1/momentum/AssignAssignSGD_1/momentumSGD_1/momentum/initial_value*
use_locking(*
T0*!
_class
loc:@SGD_1/momentum*
validate_shape(*
_output_shapes
: 
s
SGD_1/momentum/readIdentitySGD_1/momentum*!
_class
loc:@SGD_1/momentum*
_output_shapes
: *
T0
^
SGD_1/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
o
SGD_1/decay
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
Ў
SGD_1/decay/AssignAssignSGD_1/decaySGD_1/decay/initial_value*
use_locking(*
T0*
_class
loc:@SGD_1/decay*
validate_shape(*
_output_shapes
: 
j
SGD_1/decay/readIdentitySGD_1/decay*
T0*
_class
loc:@SGD_1/decay*
_output_shapes
: 

lambda_1_target_1Placeholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
t
lambda_1_sample_weights_1Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
r
loss_1/lambda_1_loss/subSublambda_1/sublambda_1_target_1*'
_output_shapes
:џџџџџџџџџ*
T0
q
loss_1/lambda_1_loss/SquareSquareloss_1/lambda_1_loss/sub*
T0*'
_output_shapes
:џџџџџџџџџ
v
+loss_1/lambda_1_loss/Mean/reduction_indicesConst*
_output_shapes
: *
valueB :
џџџџџџџџџ*
dtype0
Ж
loss_1/lambda_1_loss/MeanMeanloss_1/lambda_1_loss/Square+loss_1/lambda_1_loss/Mean/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims( 
p
-loss_1/lambda_1_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
И
loss_1/lambda_1_loss/Mean_1Meanloss_1/lambda_1_loss/Mean-loss_1/lambda_1_loss/Mean_1/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:џџџџџџџџџ

loss_1/lambda_1_loss/mulMulloss_1/lambda_1_loss/Mean_1lambda_1_sample_weights_1*
T0*#
_output_shapes
:џџџџџџџџџ
d
loss_1/lambda_1_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss_1/lambda_1_loss/NotEqualNotEquallambda_1_sample_weights_1loss_1/lambda_1_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ
}
loss_1/lambda_1_loss/CastCastloss_1/lambda_1_loss/NotEqual*

SrcT0
*#
_output_shapes
:џџџџџџџџџ*

DstT0
d
loss_1/lambda_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss_1/lambda_1_loss/Mean_2Meanloss_1/lambda_1_loss/Castloss_1/lambda_1_loss/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 

loss_1/lambda_1_loss/truedivRealDivloss_1/lambda_1_loss/mulloss_1/lambda_1_loss/Mean_2*#
_output_shapes
:џџџџџџџџџ*
T0
f
loss_1/lambda_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss_1/lambda_1_loss/Mean_3Meanloss_1/lambda_1_loss/truedivloss_1/lambda_1_loss/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
Q
loss_1/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
]

loss_1/mulMulloss_1/mul/xloss_1/lambda_1_loss/Mean_3*
_output_shapes
: *
T0
i
y_truePlaceholder*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ*
dtype0
g
maskPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
Y

loss_2/subSuby_truelambda_1/sub*
T0*'
_output_shapes
:џџџџџџџџџ
O

loss_2/AbsAbs
loss_2/sub*'
_output_shapes
:џџџџџџџџџ*
T0
R
loss_2/Less/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
`
loss_2/LessLess
loss_2/Absloss_2/Less/y*
T0*'
_output_shapes
:џџџџџџџџџ
U
loss_2/SquareSquare
loss_2/sub*
T0*'
_output_shapes
:џџџџџџџџџ
Q
loss_2/mul/xConst*
_output_shapes
: *
valueB
 *   ?*
dtype0
`

loss_2/mulMulloss_2/mul/xloss_2/Square*
T0*'
_output_shapes
:џџџџџџџџџ
Q
loss_2/Abs_1Abs
loss_2/sub*
T0*'
_output_shapes
:џџџџџџџџџ
S
loss_2/sub_1/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
c
loss_2/sub_1Subloss_2/Abs_1loss_2/sub_1/y*'
_output_shapes
:џџџџџџџџџ*
T0
S
loss_2/mul_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
c
loss_2/mul_1Mulloss_2/mul_1/xloss_2/sub_1*
T0*'
_output_shapes
:џџџџџџџџџ
p
loss_2/SelectSelectloss_2/Less
loss_2/mulloss_2/mul_1*
T0*'
_output_shapes
:џџџџџџџџџ
Z
loss_2/mul_2Mulloss_2/Selectmask*'
_output_shapes
:џџџџџџџџџ*
T0
g
loss_2/Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ


loss_2/SumSumloss_2/mul_2loss_2/Sum/reduction_indices*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims( *
T0

loss_targetPlaceholder*%
shape:џџџџџџџџџџџџџџџџџџ*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

lambda_1_target_2Placeholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
n
loss_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
t
lambda_1_sample_weights_2Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
j
'loss_3/loss_loss/Mean/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 

loss_3/loss_loss/MeanMean
loss_2/Sum'loss_3/loss_loss/Mean/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:џџџџџџџџџ
u
loss_3/loss_loss/mulMulloss_3/loss_loss/Meanloss_sample_weights*#
_output_shapes
:џџџџџџџџџ*
T0
`
loss_3/loss_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss_3/loss_loss/NotEqualNotEqualloss_sample_weightsloss_3/loss_loss/NotEqual/y*#
_output_shapes
:џџџџџџџџџ*
T0
u
loss_3/loss_loss/CastCastloss_3/loss_loss/NotEqual*

SrcT0
*#
_output_shapes
:џџџџџџџџџ*

DstT0
`
loss_3/loss_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss_3/loss_loss/Mean_1Meanloss_3/loss_loss/Castloss_3/loss_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 

loss_3/loss_loss/truedivRealDivloss_3/loss_loss/mulloss_3/loss_loss/Mean_1*
T0*#
_output_shapes
:џџџџџџџџџ
b
loss_3/loss_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss_3/loss_loss/Mean_2Meanloss_3/loss_loss/truedivloss_3/loss_loss/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Q
loss_3/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Y

loss_3/mulMulloss_3/mul/xloss_3/loss_loss/Mean_2*
T0*
_output_shapes
: 
l
loss_3/lambda_1_loss/zeros_like	ZerosLikelambda_1/sub*'
_output_shapes
:џџџџџџџџџ*
T0
u
+loss_3/lambda_1_loss/Mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
К
loss_3/lambda_1_loss/MeanMeanloss_3/lambda_1_loss/zeros_like+loss_3/lambda_1_loss/Mean/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims( 

loss_3/lambda_1_loss/mulMulloss_3/lambda_1_loss/Meanlambda_1_sample_weights_2*#
_output_shapes
:џџџџџџџџџ*
T0
d
loss_3/lambda_1_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss_3/lambda_1_loss/NotEqualNotEquallambda_1_sample_weights_2loss_3/lambda_1_loss/NotEqual/y*#
_output_shapes
:џџџџџџџџџ*
T0
}
loss_3/lambda_1_loss/CastCastloss_3/lambda_1_loss/NotEqual*#
_output_shapes
:џџџџџџџџџ*

DstT0*

SrcT0

d
loss_3/lambda_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss_3/lambda_1_loss/Mean_1Meanloss_3/lambda_1_loss/Castloss_3/lambda_1_loss/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0

loss_3/lambda_1_loss/truedivRealDivloss_3/lambda_1_loss/mulloss_3/lambda_1_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ*
T0
f
loss_3/lambda_1_loss/Const_1Const*
_output_shapes
:*
valueB: *
dtype0

loss_3/lambda_1_loss/Mean_2Meanloss_3/lambda_1_loss/truedivloss_3/lambda_1_loss/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
S
loss_3/mul_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
loss_3/mul_1Mulloss_3/mul_1/xloss_3/lambda_1_loss/Mean_2*
T0*
_output_shapes
: 
L

loss_3/addAdd
loss_3/mulloss_3/mul_1*
T0*
_output_shapes
: 
{
!metrics_2/mean_absolute_error/subSublambda_1/sublambda_1_target_2*
T0*'
_output_shapes
:џџџџџџџџџ
}
!metrics_2/mean_absolute_error/AbsAbs!metrics_2/mean_absolute_error/sub*'
_output_shapes
:џџџџџџџџџ*
T0

4metrics_2/mean_absolute_error/Mean/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ю
"metrics_2/mean_absolute_error/MeanMean!metrics_2/mean_absolute_error/Abs4metrics_2/mean_absolute_error/Mean/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims( 
m
#metrics_2/mean_absolute_error/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Г
$metrics_2/mean_absolute_error/Mean_1Mean"metrics_2/mean_absolute_error/Mean#metrics_2/mean_absolute_error/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
q
&metrics_2/mean_q/Max/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

metrics_2/mean_q/MaxMaxlambda_1/sub&metrics_2/mean_q/Max/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
	keep_dims( 
`
metrics_2/mean_q/ConstConst*
valueB: *
dtype0*
_output_shapes
:

metrics_2/mean_q/MeanMeanmetrics_2/mean_q/Maxmetrics_2/mean_q/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
[
metrics_2/mean_q/Const_1Const*
_output_shapes
: *
valueB *
dtype0

metrics_2/mean_q/Mean_1Meanmetrics_2/mean_q/Meanmetrics_2/mean_q/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 

IsVariableInitialized_29IsVariableInitializedSGD/iterations*
_output_shapes
: *!
_class
loc:@SGD/iterations*
dtype0	
y
IsVariableInitialized_30IsVariableInitializedSGD/lr*
_class
loc:@SGD/lr*
dtype0*
_output_shapes
: 

IsVariableInitialized_31IsVariableInitializedSGD/momentum*
_class
loc:@SGD/momentum*
dtype0*
_output_shapes
: 

IsVariableInitialized_32IsVariableInitialized	SGD/decay*
_output_shapes
: *
_class
loc:@SGD/decay*
dtype0

IsVariableInitialized_33IsVariableInitializedSGD_1/iterations*#
_class
loc:@SGD_1/iterations*
dtype0	*
_output_shapes
: 
}
IsVariableInitialized_34IsVariableInitializedSGD_1/lr*
_output_shapes
: *
_class
loc:@SGD_1/lr*
dtype0

IsVariableInitialized_35IsVariableInitializedSGD_1/momentum*
_output_shapes
: *!
_class
loc:@SGD_1/momentum*
dtype0

IsVariableInitialized_36IsVariableInitializedSGD_1/decay*
_class
loc:@SGD_1/decay*
dtype0*
_output_shapes
: 
И
init_1NoOp^SGD/iterations/Assign^SGD/lr/Assign^SGD/momentum/Assign^SGD/decay/Assign^SGD_1/iterations/Assign^SGD_1/lr/Assign^SGD_1/momentum/Assign^SGD_1/decay/Assign""љ
trainable_variablesсо
^
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02conv2d_1/random_uniform:0
O
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02conv2d_1/Const:0
^
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02conv2d_2/random_uniform:0
O
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:02conv2d_2/Const:0
^
conv2d_3/kernel:0conv2d_3/kernel/Assignconv2d_3/kernel/read:02conv2d_3/random_uniform:0
O
conv2d_3/bias:0conv2d_3/bias/Assignconv2d_3/bias/read:02conv2d_3/Const:0
Z
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02dense_1/random_uniform:0
K
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02dense_1/Const:0
Z
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02dense_2/random_uniform:0
K
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02dense_2/Const:0
Z
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02dense_3/random_uniform:0
K
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02dense_3/Const:0
d
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:02Adam/iterations/initial_value:0
D
	Adam/lr:0Adam/lr/AssignAdam/lr/read:02Adam/lr/initial_value:0
T
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:02Adam/beta_1/initial_value:0
T
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:02Adam/beta_2/initial_value:0
P
Adam/decay:0Adam/decay/AssignAdam/decay/read:02Adam/decay/initial_value:0
f
conv2d_1_1/kernel:0conv2d_1_1/kernel/Assignconv2d_1_1/kernel/read:02conv2d_1_1/random_uniform:0
W
conv2d_1_1/bias:0conv2d_1_1/bias/Assignconv2d_1_1/bias/read:02conv2d_1_1/Const:0
f
conv2d_2_1/kernel:0conv2d_2_1/kernel/Assignconv2d_2_1/kernel/read:02conv2d_2_1/random_uniform:0
W
conv2d_2_1/bias:0conv2d_2_1/bias/Assignconv2d_2_1/bias/read:02conv2d_2_1/Const:0
f
conv2d_3_1/kernel:0conv2d_3_1/kernel/Assignconv2d_3_1/kernel/read:02conv2d_3_1/random_uniform:0
W
conv2d_3_1/bias:0conv2d_3_1/bias/Assignconv2d_3_1/bias/read:02conv2d_3_1/Const:0
b
dense_1_1/kernel:0dense_1_1/kernel/Assigndense_1_1/kernel/read:02dense_1_1/random_uniform:0
S
dense_1_1/bias:0dense_1_1/bias/Assigndense_1_1/bias/read:02dense_1_1/Const:0
b
dense_2_1/kernel:0dense_2_1/kernel/Assigndense_2_1/kernel/read:02dense_2_1/random_uniform:0
S
dense_2_1/bias:0dense_2_1/bias/Assigndense_2_1/bias/read:02dense_2_1/Const:0
b
dense_3_1/kernel:0dense_3_1/kernel/Assigndense_3_1/kernel/read:02dense_3_1/random_uniform:0
S
dense_3_1/bias:0dense_3_1/bias/Assigndense_3_1/bias/read:02dense_3_1/Const:0
`
SGD/iterations:0SGD/iterations/AssignSGD/iterations/read:02SGD/iterations/initial_value:0
@
SGD/lr:0SGD/lr/AssignSGD/lr/read:02SGD/lr/initial_value:0
X
SGD/momentum:0SGD/momentum/AssignSGD/momentum/read:02SGD/momentum/initial_value:0
L
SGD/decay:0SGD/decay/AssignSGD/decay/read:02SGD/decay/initial_value:0
h
SGD_1/iterations:0SGD_1/iterations/AssignSGD_1/iterations/read:02 SGD_1/iterations/initial_value:0
H

SGD_1/lr:0SGD_1/lr/AssignSGD_1/lr/read:02SGD_1/lr/initial_value:0
`
SGD_1/momentum:0SGD_1/momentum/AssignSGD_1/momentum/read:02SGD_1/momentum/initial_value:0
T
SGD_1/decay:0SGD_1/decay/AssignSGD_1/decay/read:02SGD_1/decay/initial_value:0"я
	variablesсо
^
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02conv2d_1/random_uniform:0
O
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02conv2d_1/Const:0
^
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02conv2d_2/random_uniform:0
O
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:02conv2d_2/Const:0
^
conv2d_3/kernel:0conv2d_3/kernel/Assignconv2d_3/kernel/read:02conv2d_3/random_uniform:0
O
conv2d_3/bias:0conv2d_3/bias/Assignconv2d_3/bias/read:02conv2d_3/Const:0
Z
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02dense_1/random_uniform:0
K
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02dense_1/Const:0
Z
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02dense_2/random_uniform:0
K
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02dense_2/Const:0
Z
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02dense_3/random_uniform:0
K
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02dense_3/Const:0
d
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:02Adam/iterations/initial_value:0
D
	Adam/lr:0Adam/lr/AssignAdam/lr/read:02Adam/lr/initial_value:0
T
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:02Adam/beta_1/initial_value:0
T
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:02Adam/beta_2/initial_value:0
P
Adam/decay:0Adam/decay/AssignAdam/decay/read:02Adam/decay/initial_value:0
f
conv2d_1_1/kernel:0conv2d_1_1/kernel/Assignconv2d_1_1/kernel/read:02conv2d_1_1/random_uniform:0
W
conv2d_1_1/bias:0conv2d_1_1/bias/Assignconv2d_1_1/bias/read:02conv2d_1_1/Const:0
f
conv2d_2_1/kernel:0conv2d_2_1/kernel/Assignconv2d_2_1/kernel/read:02conv2d_2_1/random_uniform:0
W
conv2d_2_1/bias:0conv2d_2_1/bias/Assignconv2d_2_1/bias/read:02conv2d_2_1/Const:0
f
conv2d_3_1/kernel:0conv2d_3_1/kernel/Assignconv2d_3_1/kernel/read:02conv2d_3_1/random_uniform:0
W
conv2d_3_1/bias:0conv2d_3_1/bias/Assignconv2d_3_1/bias/read:02conv2d_3_1/Const:0
b
dense_1_1/kernel:0dense_1_1/kernel/Assigndense_1_1/kernel/read:02dense_1_1/random_uniform:0
S
dense_1_1/bias:0dense_1_1/bias/Assigndense_1_1/bias/read:02dense_1_1/Const:0
b
dense_2_1/kernel:0dense_2_1/kernel/Assigndense_2_1/kernel/read:02dense_2_1/random_uniform:0
S
dense_2_1/bias:0dense_2_1/bias/Assigndense_2_1/bias/read:02dense_2_1/Const:0
b
dense_3_1/kernel:0dense_3_1/kernel/Assigndense_3_1/kernel/read:02dense_3_1/random_uniform:0
S
dense_3_1/bias:0dense_3_1/bias/Assigndense_3_1/bias/read:02dense_3_1/Const:0
`
SGD/iterations:0SGD/iterations/AssignSGD/iterations/read:02SGD/iterations/initial_value:0
@
SGD/lr:0SGD/lr/AssignSGD/lr/read:02SGD/lr/initial_value:0
X
SGD/momentum:0SGD/momentum/AssignSGD/momentum/read:02SGD/momentum/initial_value:0
L
SGD/decay:0SGD/decay/AssignSGD/decay/read:02SGD/decay/initial_value:0
h
SGD_1/iterations:0SGD_1/iterations/AssignSGD_1/iterations/read:02 SGD_1/iterations/initial_value:0
H

SGD_1/lr:0SGD_1/lr/AssignSGD_1/lr/read:02SGD_1/lr/initial_value:0
`
SGD_1/momentum:0SGD_1/momentum/AssignSGD_1/momentum/read:02SGD_1/momentum/initial_value:0
T
SGD_1/decay:0SGD_1/decay/AssignSGD_1/decay/read:02SGD_1/decay/initial_value:0иz"       x=§	шf-ЅnзA*

episode_rewardЈЦk?`1 U$       B+M	fh-ЅnзA*

nb_episode_steps @fDџћ       ШС	шh-ЅnзA*

nb_steps @fDЧћъе$       B+M	FЭІnзA*

episode_reward7a?k^F&       sOу 	`ЮІnзA*

nb_episode_steps @\D;Ьoc       йм2	тЮІnзA*

nb_steps @сDџЪ3Є$       B+M	ЬьЇnзA*

episode_reward=
W?ь-:і&       sOу 	ъэЇnзA*

nb_episode_steps  RD_Ђ+       йм2	lюЇnзA*

nb_steps  %E§Н$       B+M	Ъ7вЇnзA*

episode_reward/нD?Д3*&       sOу 	ы8вЇnзA*

nb_episode_steps @@DѕwН       йм2	m9вЇnзA*

nb_steps 0UEу$       B+M	ЮOІЈnзA*

episode_rewardјSC?&ЧtЦ&       sOу 	єPІЈnзA*

nb_episode_steps Р>DQдХ       йм2	{QІЈnзA*

nb_steps pEйЮъ$       B+M	ЬЉnзA*

episode_rewardd;_?н&       sOу 	ђЉnзA*

nb_episode_steps  ZDэюЧИ       йм2	tЉnзA*

nb_steps АEHИ$       B+M	AЗQЊnзA*

episode_rewardHс:?Е9k&       sOу 	[ИQЊnзA*

nb_episode_steps 6DњёПЗ       йм2	сИQЊnзA*

nb_steps ДEW2Ђ'$       B+M	ПИ1ЋnзA*

episode_rewardjМT?№&       sOу 	иЙ1ЋnзA*

nb_episode_steps РODDзjё       йм2	ZК1ЋnзA*

nb_steps xЮE@ы$       B+M	Ям	ЌnзA*

episode_rewardВO? m/&       sOу 	№н	ЌnзA*

nb_episode_steps РJDяы5a       йм2	sо	ЌnзA*

nb_steps ачE#ФММ$       B+M	kA­nзA	*

episode_reward)\o?Г-чЬ&       sOу 	B­nзA	*

nb_episode_steps РiDdШбЄ       йм2	C­nзA	*

nb_steps F:КЖљ$       B+M	.и­nзA
*

episode_rewardЫЁE?Ьjє-&       sOу 	D/и­nзA
*

nb_episode_steps  ADP{ПK       йм2	Ъ/и­nзA
*

nb_steps Fх@ФY$       B+M	cAЎnзA*

episode_rewardP7?XЋжё&       sOу 	BЎnзA*

nb_episode_steps @3DBiЇш       йм2	ўBЎnзA*

nb_steps ШF;[$       B+M	V}іАnзA*

episode_reward`хP?Пћ &       sOу 	~іАnзA*

nb_episode_steps  LDБиш?       йм2	іАnзA*

nb_steps &FЭ]еK$       B+M	7pГnзA*

episode_rewardыQ?zI
&       sOу 	Ц8pГnзA*

nb_episode_steps  MD5/       йм2	H9pГnзA*

nb_steps X3FuЕ6$       B+M	-ЎЖnзA*

episode_rewardлљ^?М}&       sOу 	`ЏЖnзA*

nb_episode_steps РYD)1р       йм2	тЏЖnзA*

nb_steps є@Fbo'О$       B+M	vЗnзA*

episode_rewardТѕ>Ыж­Ч&       sOу 	4vЗnзA*

nb_episode_steps  №CїYЋД       йм2	КvЗnзA*

nb_steps tHF|іЄь