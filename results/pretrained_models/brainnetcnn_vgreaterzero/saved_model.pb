ã
¢ø
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-0-g3f878cff5b68£

e2e_conv_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namee2e_conv_12/kernel

&e2e_conv_12/kernel/Read/ReadVariableOpReadVariableOpe2e_conv_12/kernel*&
_output_shapes
: *
dtype0

e2e_conv_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *#
shared_namee2e_conv_13/kernel

&e2e_conv_13/kernel/Read/ReadVariableOpReadVariableOpe2e_conv_13/kernel*&
_output_shapes
:  *
dtype0

Edge-to-Node/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*$
shared_nameEdge-to-Node/kernel

'Edge-to-Node/kernel/Read/ReadVariableOpReadVariableOpEdge-to-Node/kernel*&
_output_shapes
: @*
dtype0
z
Edge-to-Node/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameEdge-to-Node/bias
s
%Edge-to-Node/bias/Read/ReadVariableOpReadVariableOpEdge-to-Node/bias*
_output_shapes
:@*
dtype0

Node-to-Graph/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*%
shared_nameNode-to-Graph/kernel

(Node-to-Graph/kernel/Read/ReadVariableOpReadVariableOpNode-to-Graph/kernel*&
_output_shapes
:@@*
dtype0
|
Node-to-Graph/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameNode-to-Graph/bias
u
&Node-to-Graph/bias/Read/ReadVariableOpReadVariableOpNode-to-Graph/bias*
_output_shapes
:@*
dtype0
{
dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	C* 
shared_namedense_28/kernel
t
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel*
_output_shapes
:	C*
dtype0
s
dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_28/bias
l
!dense_28/bias/Read/ReadVariableOpReadVariableOpdense_28/bias*
_output_shapes	
:*
dtype0
|
dense_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_29/kernel
u
#dense_29/kernel/Read/ReadVariableOpReadVariableOpdense_29/kernel* 
_output_shapes
:
*
dtype0
s
dense_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_29/bias
l
!dense_29/bias/Read/ReadVariableOpReadVariableOpdense_29/bias*
_output_shapes	
:*
dtype0
{
dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_30/kernel
t
#dense_30/kernel/Read/ReadVariableOpReadVariableOpdense_30/kernel*
_output_shapes
:	*
dtype0
r
dense_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_30/bias
k
!dense_30/bias/Read/ReadVariableOpReadVariableOpdense_30/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
n
accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator
g
accumulator/Read/ReadVariableOpReadVariableOpaccumulator*
_output_shapes
:*
dtype0

Adam/e2e_conv_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/e2e_conv_12/kernel/m

-Adam/e2e_conv_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/e2e_conv_12/kernel/m*&
_output_shapes
: *
dtype0

Adam/e2e_conv_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  **
shared_nameAdam/e2e_conv_13/kernel/m

-Adam/e2e_conv_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/e2e_conv_13/kernel/m*&
_output_shapes
:  *
dtype0

Adam/Edge-to-Node/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameAdam/Edge-to-Node/kernel/m

.Adam/Edge-to-Node/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Edge-to-Node/kernel/m*&
_output_shapes
: @*
dtype0

Adam/Edge-to-Node/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/Edge-to-Node/bias/m

,Adam/Edge-to-Node/bias/m/Read/ReadVariableOpReadVariableOpAdam/Edge-to-Node/bias/m*
_output_shapes
:@*
dtype0

Adam/Node-to-Graph/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*,
shared_nameAdam/Node-to-Graph/kernel/m

/Adam/Node-to-Graph/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Node-to-Graph/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/Node-to-Graph/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameAdam/Node-to-Graph/bias/m

-Adam/Node-to-Graph/bias/m/Read/ReadVariableOpReadVariableOpAdam/Node-to-Graph/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	C*'
shared_nameAdam/dense_28/kernel/m

*Adam/dense_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/m*
_output_shapes
:	C*
dtype0

Adam/dense_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_28/bias/m
z
(Adam/dense_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_29/kernel/m

*Adam/dense_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_29/bias/m
z
(Adam/dense_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_30/kernel/m

*Adam/dense_30/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/m*
_output_shapes
:	*
dtype0

Adam/dense_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_30/bias/m
y
(Adam/dense_30/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/m*
_output_shapes
:*
dtype0

Adam/e2e_conv_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/e2e_conv_12/kernel/v

-Adam/e2e_conv_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/e2e_conv_12/kernel/v*&
_output_shapes
: *
dtype0

Adam/e2e_conv_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  **
shared_nameAdam/e2e_conv_13/kernel/v

-Adam/e2e_conv_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/e2e_conv_13/kernel/v*&
_output_shapes
:  *
dtype0

Adam/Edge-to-Node/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameAdam/Edge-to-Node/kernel/v

.Adam/Edge-to-Node/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Edge-to-Node/kernel/v*&
_output_shapes
: @*
dtype0

Adam/Edge-to-Node/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/Edge-to-Node/bias/v

,Adam/Edge-to-Node/bias/v/Read/ReadVariableOpReadVariableOpAdam/Edge-to-Node/bias/v*
_output_shapes
:@*
dtype0

Adam/Node-to-Graph/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*,
shared_nameAdam/Node-to-Graph/kernel/v

/Adam/Node-to-Graph/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Node-to-Graph/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/Node-to-Graph/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameAdam/Node-to-Graph/bias/v

-Adam/Node-to-Graph/bias/v/Read/ReadVariableOpReadVariableOpAdam/Node-to-Graph/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	C*'
shared_nameAdam/dense_28/kernel/v

*Adam/dense_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/v*
_output_shapes
:	C*
dtype0

Adam/dense_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_28/bias/v
z
(Adam/dense_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_29/kernel/v

*Adam/dense_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_29/bias/v
z
(Adam/dense_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_30/kernel/v

*Adam/dense_30/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/v*
_output_shapes
:	*
dtype0

Adam/dense_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_30/bias/v
y
(Adam/dense_30/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
¹h
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ôg
valueêgBçg Bàg
¼
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 


kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*


kernel
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses*
¦

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses*
¦

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses*
¥
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:_random_generator
;__call__
*<&call_and_return_all_conditional_losses* 

=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses* 
* 

C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses* 
¦

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses*
¥
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U_random_generator
V__call__
*W&call_and_return_all_conditional_losses* 
¦

Xkernel
Ybias
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses*
¥
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d_random_generator
e__call__
*f&call_and_return_all_conditional_losses* 
¦

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses*
´
oiter

pbeta_1

qbeta_2
	rdecay
slearning_ratemÊmË&mÌ'mÍ.mÎ/mÏImÐJmÑXmÒYmÓgmÔhmÕvÖv×&vØ'vÙ.vÚ/vÛIvÜJvÝXvÞYvßgvàhvá*
Z
0
1
&2
'3
.4
/5
I6
J7
X8
Y9
g10
h11*
Z
0
1
&2
'3
.4
/5
I6
J7
X8
Y9
g10
h11*

t0
u1
v2
w3* 
°
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

}serving_default* 
b\
VARIABLE_VALUEe2e_conv_12/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
	
t0* 

~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
b\
VARIABLE_VALUEe2e_conv_13/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
	
u0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUEEdge-to-Node/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEEdge-to-Node/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1*

&0
'1*
	
v0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*
* 
* 
d^
VARIABLE_VALUENode-to-Graph/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUENode-to-Graph/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

.0
/1*

.0
/1*
	
w0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
6	variables
7trainable_variables
8regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_28/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_28/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

I0
J1*

I0
J1*
* 

¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses* 
* 
* 
* 
_Y
VARIABLE_VALUEdense_29/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_29/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

X0
Y1*

X0
Y1*
* 

«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
`	variables
atrainable_variables
bregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses* 
* 
* 
* 
_Y
VARIABLE_VALUEdense_30/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_30/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

g0
h1*

g0
h1*
* 

µnon_trainable_variables
¶layers
·metrics
 ¸layer_regularization_losses
¹layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
j
0
1
2
3
4
5
6
7
	8

9
10
11
12
13*

º0
»1
¼2*
* 
* 
* 
* 
* 
* 
	
t0* 
* 
* 
* 
* 
	
u0* 
* 
* 
* 
* 
	
v0* 
* 
* 
* 
* 
	
w0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

½total

¾count
¿	variables
À	keras_api*
M

Átotal

Âcount
Ã
_fn_kwargs
Ä	variables
Å	keras_api*
G
Æ
thresholds
Çaccumulator
È	variables
É	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

½0
¾1*

¿	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

Á0
Â1*

Ä	variables*
* 
_Y
VARIABLE_VALUEaccumulator:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUE*

Ç0*

È	variables*

VARIABLE_VALUEAdam/e2e_conv_12/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/e2e_conv_13/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/Edge-to-Node/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/Edge-to-Node/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/Node-to-Graph/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/Node-to-Graph/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_28/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_28/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_29/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_29/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_30/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_30/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/e2e_conv_12/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/e2e_conv_13/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/Edge-to-Node/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/Edge-to-Node/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/Node-to-Graph/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/Node-to-Graph/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_28/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_28/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_29/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_29/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_30/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_30/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_imgPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
~
serving_default_input_strucPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Ä
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_imgserving_default_input_struce2e_conv_12/kernele2e_conv_13/kernelEdge-to-Node/kernelEdge-to-Node/biasNode-to-Graph/kernelNode-to-Graph/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/biasdense_30/kerneldense_30/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_116070
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&e2e_conv_12/kernel/Read/ReadVariableOp&e2e_conv_13/kernel/Read/ReadVariableOp'Edge-to-Node/kernel/Read/ReadVariableOp%Edge-to-Node/bias/Read/ReadVariableOp(Node-to-Graph/kernel/Read/ReadVariableOp&Node-to-Graph/bias/Read/ReadVariableOp#dense_28/kernel/Read/ReadVariableOp!dense_28/bias/Read/ReadVariableOp#dense_29/kernel/Read/ReadVariableOp!dense_29/bias/Read/ReadVariableOp#dense_30/kernel/Read/ReadVariableOp!dense_30/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpaccumulator/Read/ReadVariableOp-Adam/e2e_conv_12/kernel/m/Read/ReadVariableOp-Adam/e2e_conv_13/kernel/m/Read/ReadVariableOp.Adam/Edge-to-Node/kernel/m/Read/ReadVariableOp,Adam/Edge-to-Node/bias/m/Read/ReadVariableOp/Adam/Node-to-Graph/kernel/m/Read/ReadVariableOp-Adam/Node-to-Graph/bias/m/Read/ReadVariableOp*Adam/dense_28/kernel/m/Read/ReadVariableOp(Adam/dense_28/bias/m/Read/ReadVariableOp*Adam/dense_29/kernel/m/Read/ReadVariableOp(Adam/dense_29/bias/m/Read/ReadVariableOp*Adam/dense_30/kernel/m/Read/ReadVariableOp(Adam/dense_30/bias/m/Read/ReadVariableOp-Adam/e2e_conv_12/kernel/v/Read/ReadVariableOp-Adam/e2e_conv_13/kernel/v/Read/ReadVariableOp.Adam/Edge-to-Node/kernel/v/Read/ReadVariableOp,Adam/Edge-to-Node/bias/v/Read/ReadVariableOp/Adam/Node-to-Graph/kernel/v/Read/ReadVariableOp-Adam/Node-to-Graph/bias/v/Read/ReadVariableOp*Adam/dense_28/kernel/v/Read/ReadVariableOp(Adam/dense_28/bias/v/Read/ReadVariableOp*Adam/dense_29/kernel/v/Read/ReadVariableOp(Adam/dense_29/bias/v/Read/ReadVariableOp*Adam/dense_30/kernel/v/Read/ReadVariableOp(Adam/dense_30/bias/v/Read/ReadVariableOpConst*;
Tin4
220	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_116599
ì	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamee2e_conv_12/kernele2e_conv_13/kernelEdge-to-Node/kernelEdge-to-Node/biasNode-to-Graph/kernelNode-to-Graph/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/biasdense_30/kerneldense_30/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1accumulatorAdam/e2e_conv_12/kernel/mAdam/e2e_conv_13/kernel/mAdam/Edge-to-Node/kernel/mAdam/Edge-to-Node/bias/mAdam/Node-to-Graph/kernel/mAdam/Node-to-Graph/bias/mAdam/dense_28/kernel/mAdam/dense_28/bias/mAdam/dense_29/kernel/mAdam/dense_29/bias/mAdam/dense_30/kernel/mAdam/dense_30/bias/mAdam/e2e_conv_12/kernel/vAdam/e2e_conv_13/kernel/vAdam/Edge-to-Node/kernel/vAdam/Edge-to-Node/bias/vAdam/Node-to-Graph/kernel/vAdam/Node-to-Graph/bias/vAdam/dense_28/kernel/vAdam/dense_28/bias/vAdam/dense_29/kernel/vAdam/dense_29/bias/vAdam/dense_30/kernel/vAdam/dense_30/bias/v*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_116747ð³
µ%
ô
G__inference_e2e_conv_12_layer_call_and_return_conditional_losses_115027

inputs1
readvariableop_resource: 
identity¢ReadVariableOp¢ReadVariableOp_1¢4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOpn
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
: *
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*"
_output_shapes
: *

begin_mask*
end_mask*
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             s
ReshapeReshapestrided_slice:output:0Reshape/shape:output:0*
T0*&
_output_shapes
: p
ReadVariableOp_1ReadVariableOpreadvariableop_resource*&
_output_shapes
: *
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*"
_output_shapes
: *

begin_mask*
end_mask*
shrink_axis_maskh
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             y
	Reshape_1Reshapestrided_slice_1:output:0Reshape_1/shape:output:0*
T0*&
_output_shapes
: 
convolutionConv2DinputsReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

convolution_1Conv2DinputsReshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :«
concatConcatV2convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0concat/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_1ConcatV2convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0concat_1/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
addAddV2concat:output:0concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
: *
dtype0
%e2e_conv_12/kernel/Regularizer/SquareSquare<e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$e2e_conv_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¤
"e2e_conv_12/kernel/Regularizer/SumSum)e2e_conv_12/kernel/Regularizer/Square:y:0-e2e_conv_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$e2e_conv_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¦
"e2e_conv_12/kernel/Regularizer/mulMul-e2e_conv_12/kernel/Regularizer/mul/x:output:0+e2e_conv_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
NoOpNoOp^ReadVariableOp^ReadVariableOp_15^e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12l
4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§

ø
D__inference_dense_29_layer_call_and_return_conditional_losses_115174

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
»
I__inference_Node-to-Graph_layer_call_and_return_conditional_losses_115109

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0¢
'Node-to-Graph/kernel/Regularizer/SquareSquare>Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@
&Node-to-Graph/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ª
$Node-to-Graph/kernel/Regularizer/SumSum+Node-to-Graph/kernel/Regularizer/Square:y:0/Node-to-Graph/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&Node-to-Graph/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¬
$Node-to-Graph/kernel/Regularizer/mulMul/Node-to-Graph/kernel/Regularizer/mul/x:output:0-Node-to-Graph/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp7^Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2p
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
§
ß
)__inference_model_16_layer_call_fn_115779
inputs_0
inputs_1!
unknown: #
	unknown_0:  #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	C
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_16_layer_call_and_return_conditional_losses_115502o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ü	
e
F__inference_dropout_49_layer_call_and_return_conditional_losses_116326

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À

D__inference_model_16_layer_call_and_return_conditional_losses_115898
inputs_0
inputs_1=
#e2e_conv_12_readvariableop_resource: =
#e2e_conv_13_readvariableop_resource:  E
+edge_to_node_conv2d_readvariableop_resource: @:
,edge_to_node_biasadd_readvariableop_resource:@F
,node_to_graph_conv2d_readvariableop_resource:@@;
-node_to_graph_biasadd_readvariableop_resource:@:
'dense_28_matmul_readvariableop_resource:	C7
(dense_28_biasadd_readvariableop_resource:	;
'dense_29_matmul_readvariableop_resource:
7
(dense_29_biasadd_readvariableop_resource:	:
'dense_30_matmul_readvariableop_resource:	6
(dense_30_biasadd_readvariableop_resource:
identity¢#Edge-to-Node/BiasAdd/ReadVariableOp¢"Edge-to-Node/Conv2D/ReadVariableOp¢5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp¢$Node-to-Graph/BiasAdd/ReadVariableOp¢#Node-to-Graph/Conv2D/ReadVariableOp¢6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp¢dense_28/BiasAdd/ReadVariableOp¢dense_28/MatMul/ReadVariableOp¢dense_29/BiasAdd/ReadVariableOp¢dense_29/MatMul/ReadVariableOp¢dense_30/BiasAdd/ReadVariableOp¢dense_30/MatMul/ReadVariableOp¢e2e_conv_12/ReadVariableOp¢e2e_conv_12/ReadVariableOp_1¢4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp¢e2e_conv_13/ReadVariableOp¢e2e_conv_13/ReadVariableOp_1¢4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp
e2e_conv_12/ReadVariableOpReadVariableOp#e2e_conv_12_readvariableop_resource*&
_output_shapes
: *
dtype0p
e2e_conv_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!e2e_conv_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!e2e_conv_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ã
e2e_conv_12/strided_sliceStridedSlice"e2e_conv_12/ReadVariableOp:value:0(e2e_conv_12/strided_slice/stack:output:0*e2e_conv_12/strided_slice/stack_1:output:0*e2e_conv_12/strided_slice/stack_2:output:0*
Index0*
T0*"
_output_shapes
: *

begin_mask*
end_mask*
shrink_axis_maskr
e2e_conv_12/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             
e2e_conv_12/ReshapeReshape"e2e_conv_12/strided_slice:output:0"e2e_conv_12/Reshape/shape:output:0*
T0*&
_output_shapes
: 
e2e_conv_12/ReadVariableOp_1ReadVariableOp#e2e_conv_12_readvariableop_resource*&
_output_shapes
: *
dtype0r
!e2e_conv_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#e2e_conv_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#e2e_conv_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Í
e2e_conv_12/strided_slice_1StridedSlice$e2e_conv_12/ReadVariableOp_1:value:0*e2e_conv_12/strided_slice_1/stack:output:0,e2e_conv_12/strided_slice_1/stack_1:output:0,e2e_conv_12/strided_slice_1/stack_2:output:0*
Index0*
T0*"
_output_shapes
: *

begin_mask*
end_mask*
shrink_axis_maskt
e2e_conv_12/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             
e2e_conv_12/Reshape_1Reshape$e2e_conv_12/strided_slice_1:output:0$e2e_conv_12/Reshape_1/shape:output:0*
T0*&
_output_shapes
: ¬
e2e_conv_12/convolutionConv2Dinputs_0e2e_conv_12/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
°
e2e_conv_12/convolution_1Conv2Dinputs_0e2e_conv_12/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
Y
e2e_conv_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :£
e2e_conv_12/concatConcatV2"e2e_conv_12/convolution_1:output:0"e2e_conv_12/convolution_1:output:0"e2e_conv_12/convolution_1:output:0"e2e_conv_12/convolution_1:output:0"e2e_conv_12/convolution_1:output:0"e2e_conv_12/convolution_1:output:0"e2e_conv_12/convolution_1:output:0"e2e_conv_12/convolution_1:output:0 e2e_conv_12/concat/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
e2e_conv_12/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
e2e_conv_12/concat_1ConcatV2 e2e_conv_12/convolution:output:0 e2e_conv_12/convolution:output:0 e2e_conv_12/convolution:output:0 e2e_conv_12/convolution:output:0 e2e_conv_12/convolution:output:0 e2e_conv_12/convolution:output:0 e2e_conv_12/convolution:output:0 e2e_conv_12/convolution:output:0"e2e_conv_12/concat_1/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
e2e_conv_12/addAddV2e2e_conv_12/concat:output:0e2e_conv_12/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
e2e_conv_13/ReadVariableOpReadVariableOp#e2e_conv_13_readvariableop_resource*&
_output_shapes
:  *
dtype0p
e2e_conv_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!e2e_conv_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!e2e_conv_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ã
e2e_conv_13/strided_sliceStridedSlice"e2e_conv_13/ReadVariableOp:value:0(e2e_conv_13/strided_slice/stack:output:0*e2e_conv_13/strided_slice/stack_1:output:0*e2e_conv_13/strided_slice/stack_2:output:0*
Index0*
T0*"
_output_shapes
:  *

begin_mask*
end_mask*
shrink_axis_maskr
e2e_conv_13/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              
e2e_conv_13/ReshapeReshape"e2e_conv_13/strided_slice:output:0"e2e_conv_13/Reshape/shape:output:0*
T0*&
_output_shapes
:  
e2e_conv_13/ReadVariableOp_1ReadVariableOp#e2e_conv_13_readvariableop_resource*&
_output_shapes
:  *
dtype0r
!e2e_conv_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#e2e_conv_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#e2e_conv_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Í
e2e_conv_13/strided_slice_1StridedSlice$e2e_conv_13/ReadVariableOp_1:value:0*e2e_conv_13/strided_slice_1/stack:output:0,e2e_conv_13/strided_slice_1/stack_1:output:0,e2e_conv_13/strided_slice_1/stack_2:output:0*
Index0*
T0*"
_output_shapes
:  *

begin_mask*
end_mask*
shrink_axis_maskt
e2e_conv_13/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              
e2e_conv_13/Reshape_1Reshape$e2e_conv_13/strided_slice_1:output:0$e2e_conv_13/Reshape_1/shape:output:0*
T0*&
_output_shapes
:  ·
e2e_conv_13/convolutionConv2De2e_conv_12/add:z:0e2e_conv_13/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
»
e2e_conv_13/convolution_1Conv2De2e_conv_12/add:z:0e2e_conv_13/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
Y
e2e_conv_13/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :£
e2e_conv_13/concatConcatV2"e2e_conv_13/convolution_1:output:0"e2e_conv_13/convolution_1:output:0"e2e_conv_13/convolution_1:output:0"e2e_conv_13/convolution_1:output:0"e2e_conv_13/convolution_1:output:0"e2e_conv_13/convolution_1:output:0"e2e_conv_13/convolution_1:output:0"e2e_conv_13/convolution_1:output:0 e2e_conv_13/concat/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
e2e_conv_13/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
e2e_conv_13/concat_1ConcatV2 e2e_conv_13/convolution:output:0 e2e_conv_13/convolution:output:0 e2e_conv_13/convolution:output:0 e2e_conv_13/convolution:output:0 e2e_conv_13/convolution:output:0 e2e_conv_13/convolution:output:0 e2e_conv_13/convolution:output:0 e2e_conv_13/convolution:output:0"e2e_conv_13/concat_1/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
e2e_conv_13/addAddV2e2e_conv_13/concat:output:0e2e_conv_13/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"Edge-to-Node/Conv2D/ReadVariableOpReadVariableOp+edge_to_node_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Á
Edge-to-Node/Conv2DConv2De2e_conv_13/add:z:0*Edge-to-Node/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

#Edge-to-Node/BiasAdd/ReadVariableOpReadVariableOp,edge_to_node_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¤
Edge-to-Node/BiasAddBiasAddEdge-to-Node/Conv2D:output:0+Edge-to-Node/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
Edge-to-Node/ReluReluEdge-to-Node/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
#Node-to-Graph/Conv2D/ReadVariableOpReadVariableOp,node_to_graph_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ï
Node-to-Graph/Conv2DConv2DEdge-to-Node/Relu:activations:0+Node-to-Graph/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

$Node-to-Graph/BiasAdd/ReadVariableOpReadVariableOp-node_to_graph_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0§
Node-to-Graph/BiasAddBiasAddNode-to-Graph/Conv2D:output:0,Node-to-Graph/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
Node-to-Graph/ReluReluNode-to-Graph/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
dropout_48/IdentityIdentity Node-to-Graph/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
flatten_16/ReshapeReshapedropout_48/Identity:output:0flatten_16/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¦
concatenate_2/concatConcatV2flatten_16/Reshape:output:0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes
:	C*
dtype0
dense_28/MatMulMatMulconcatenate_2/concat:output:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout_49/IdentityIdentitydense_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_29/MatMulMatMuldropout_49/Identity:output:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_29/ReluReludense_29/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout_50/IdentityIdentitydense_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_30/MatMulMatMuldropout_50/Identity:output:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_30/SigmoidSigmoiddense_30/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp#e2e_conv_12_readvariableop_resource*&
_output_shapes
: *
dtype0
%e2e_conv_12/kernel/Regularizer/SquareSquare<e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$e2e_conv_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¤
"e2e_conv_12/kernel/Regularizer/SumSum)e2e_conv_12/kernel/Regularizer/Square:y:0-e2e_conv_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$e2e_conv_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¦
"e2e_conv_12/kernel/Regularizer/mulMul-e2e_conv_12/kernel/Regularizer/mul/x:output:0+e2e_conv_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
:  
4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp#e2e_conv_13_readvariableop_resource*&
_output_shapes
:  *
dtype0
%e2e_conv_13/kernel/Regularizer/SquareSquare<e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  }
$e2e_conv_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¤
"e2e_conv_13/kernel/Regularizer/SumSum)e2e_conv_13/kernel/Regularizer/Square:y:0-e2e_conv_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$e2e_conv_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¦
"e2e_conv_13/kernel/Regularizer/mulMul-e2e_conv_13/kernel/Regularizer/mul/x:output:0+e2e_conv_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ©
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+edge_to_node_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0 
&Edge-to-Node/kernel/Regularizer/SquareSquare=Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @~
%Edge-to-Node/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             §
#Edge-to-Node/kernel/Regularizer/SumSum*Edge-to-Node/kernel/Regularizer/Square:y:0.Edge-to-Node/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%Edge-to-Node/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#Edge-to-Node/kernel/Regularizer/mulMul.Edge-to-Node/kernel/Regularizer/mul/x:output:0,Edge-to-Node/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: «
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOpReadVariableOp,node_to_graph_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0¢
'Node-to-Graph/kernel/Regularizer/SquareSquare>Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@
&Node-to-Graph/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ª
$Node-to-Graph/kernel/Regularizer/SumSum+Node-to-Graph/kernel/Regularizer/Square:y:0/Node-to-Graph/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&Node-to-Graph/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¬
$Node-to-Graph/kernel/Regularizer/mulMul/Node-to-Graph/kernel/Regularizer/mul/x:output:0-Node-to-Graph/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
IdentityIdentitydense_30/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ
NoOpNoOp$^Edge-to-Node/BiasAdd/ReadVariableOp#^Edge-to-Node/Conv2D/ReadVariableOp6^Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp%^Node-to-Graph/BiasAdd/ReadVariableOp$^Node-to-Graph/Conv2D/ReadVariableOp7^Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp^e2e_conv_12/ReadVariableOp^e2e_conv_12/ReadVariableOp_15^e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp^e2e_conv_13/ReadVariableOp^e2e_conv_13/ReadVariableOp_15^e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2J
#Edge-to-Node/BiasAdd/ReadVariableOp#Edge-to-Node/BiasAdd/ReadVariableOp2H
"Edge-to-Node/Conv2D/ReadVariableOp"Edge-to-Node/Conv2D/ReadVariableOp2n
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp2L
$Node-to-Graph/BiasAdd/ReadVariableOp$Node-to-Graph/BiasAdd/ReadVariableOp2J
#Node-to-Graph/Conv2D/ReadVariableOp#Node-to-Graph/Conv2D/ReadVariableOp2p
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp28
e2e_conv_12/ReadVariableOpe2e_conv_12/ReadVariableOp2<
e2e_conv_12/ReadVariableOp_1e2e_conv_12/ReadVariableOp_12l
4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp28
e2e_conv_13/ReadVariableOpe2e_conv_13/ReadVariableOp2<
e2e_conv_13/ReadVariableOp_1e2e_conv_13/ReadVariableOp_12l
4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
µ%
ô
G__inference_e2e_conv_13_layer_call_and_return_conditional_losses_116164

inputs1
readvariableop_resource:  
identity¢ReadVariableOp¢ReadVariableOp_1¢4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOpn
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:  *
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*"
_output_shapes
:  *

begin_mask*
end_mask*
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              s
ReshapeReshapestrided_slice:output:0Reshape/shape:output:0*
T0*&
_output_shapes
:  p
ReadVariableOp_1ReadVariableOpreadvariableop_resource*&
_output_shapes
:  *
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*"
_output_shapes
:  *

begin_mask*
end_mask*
shrink_axis_maskh
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              y
	Reshape_1Reshapestrided_slice_1:output:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:  
convolutionConv2DinputsReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

convolution_1Conv2DinputsReshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :«
concatConcatV2convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0concat/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_1ConcatV2convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0concat_1/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
addAddV2concat:output:0concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:  *
dtype0
%e2e_conv_13/kernel/Regularizer/SquareSquare<e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  }
$e2e_conv_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¤
"e2e_conv_13/kernel/Regularizer/SumSum)e2e_conv_13/kernel/Regularizer/Square:y:0-e2e_conv_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$e2e_conv_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¦
"e2e_conv_13/kernel/Regularizer/mulMul-e2e_conv_13/kernel/Regularizer/mul/x:output:0+e2e_conv_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
NoOpNoOp^ReadVariableOp^ReadVariableOp_15^e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12l
4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ý
d
F__inference_dropout_49_layer_call_and_return_conditional_losses_116314

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á
G
+__inference_dropout_48_layer_call_fn_116233

inputs
identity¹
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_48_layer_call_and_return_conditional_losses_115120h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
«_
¦
__inference__traced_save_116599
file_prefix1
-savev2_e2e_conv_12_kernel_read_readvariableop1
-savev2_e2e_conv_13_kernel_read_readvariableop2
.savev2_edge_to_node_kernel_read_readvariableop0
,savev2_edge_to_node_bias_read_readvariableop3
/savev2_node_to_graph_kernel_read_readvariableop1
-savev2_node_to_graph_bias_read_readvariableop.
*savev2_dense_28_kernel_read_readvariableop,
(savev2_dense_28_bias_read_readvariableop.
*savev2_dense_29_kernel_read_readvariableop,
(savev2_dense_29_bias_read_readvariableop.
*savev2_dense_30_kernel_read_readvariableop,
(savev2_dense_30_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop*
&savev2_accumulator_read_readvariableop8
4savev2_adam_e2e_conv_12_kernel_m_read_readvariableop8
4savev2_adam_e2e_conv_13_kernel_m_read_readvariableop9
5savev2_adam_edge_to_node_kernel_m_read_readvariableop7
3savev2_adam_edge_to_node_bias_m_read_readvariableop:
6savev2_adam_node_to_graph_kernel_m_read_readvariableop8
4savev2_adam_node_to_graph_bias_m_read_readvariableop5
1savev2_adam_dense_28_kernel_m_read_readvariableop3
/savev2_adam_dense_28_bias_m_read_readvariableop5
1savev2_adam_dense_29_kernel_m_read_readvariableop3
/savev2_adam_dense_29_bias_m_read_readvariableop5
1savev2_adam_dense_30_kernel_m_read_readvariableop3
/savev2_adam_dense_30_bias_m_read_readvariableop8
4savev2_adam_e2e_conv_12_kernel_v_read_readvariableop8
4savev2_adam_e2e_conv_13_kernel_v_read_readvariableop9
5savev2_adam_edge_to_node_kernel_v_read_readvariableop7
3savev2_adam_edge_to_node_bias_v_read_readvariableop:
6savev2_adam_node_to_graph_kernel_v_read_readvariableop8
4savev2_adam_node_to_graph_bias_v_read_readvariableop5
1savev2_adam_dense_28_kernel_v_read_readvariableop3
/savev2_adam_dense_28_bias_v_read_readvariableop5
1savev2_adam_dense_29_kernel_v_read_readvariableop3
/savev2_adam_dense_29_bias_v_read_readvariableop5
1savev2_adam_dense_30_kernel_v_read_readvariableop3
/savev2_adam_dense_30_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: å
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*
valueB/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHË
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B à
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_e2e_conv_12_kernel_read_readvariableop-savev2_e2e_conv_13_kernel_read_readvariableop.savev2_edge_to_node_kernel_read_readvariableop,savev2_edge_to_node_bias_read_readvariableop/savev2_node_to_graph_kernel_read_readvariableop-savev2_node_to_graph_bias_read_readvariableop*savev2_dense_28_kernel_read_readvariableop(savev2_dense_28_bias_read_readvariableop*savev2_dense_29_kernel_read_readvariableop(savev2_dense_29_bias_read_readvariableop*savev2_dense_30_kernel_read_readvariableop(savev2_dense_30_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop&savev2_accumulator_read_readvariableop4savev2_adam_e2e_conv_12_kernel_m_read_readvariableop4savev2_adam_e2e_conv_13_kernel_m_read_readvariableop5savev2_adam_edge_to_node_kernel_m_read_readvariableop3savev2_adam_edge_to_node_bias_m_read_readvariableop6savev2_adam_node_to_graph_kernel_m_read_readvariableop4savev2_adam_node_to_graph_bias_m_read_readvariableop1savev2_adam_dense_28_kernel_m_read_readvariableop/savev2_adam_dense_28_bias_m_read_readvariableop1savev2_adam_dense_29_kernel_m_read_readvariableop/savev2_adam_dense_29_bias_m_read_readvariableop1savev2_adam_dense_30_kernel_m_read_readvariableop/savev2_adam_dense_30_bias_m_read_readvariableop4savev2_adam_e2e_conv_12_kernel_v_read_readvariableop4savev2_adam_e2e_conv_13_kernel_v_read_readvariableop5savev2_adam_edge_to_node_kernel_v_read_readvariableop3savev2_adam_edge_to_node_bias_v_read_readvariableop6savev2_adam_node_to_graph_kernel_v_read_readvariableop4savev2_adam_node_to_graph_bias_v_read_readvariableop1savev2_adam_dense_28_kernel_v_read_readvariableop/savev2_adam_dense_28_bias_v_read_readvariableop1savev2_adam_dense_29_kernel_v_read_readvariableop/savev2_adam_dense_29_bias_v_read_readvariableop1savev2_adam_dense_30_kernel_v_read_readvariableop/savev2_adam_dense_30_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *=
dtypes3
12/	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ï
_input_shapes½
º: : :  : @:@:@@:@:	C::
::	:: : : : : : : : : :: :  : @:@:@@:@:	C::
::	:: :  : @:@:@@:@:	C::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: :,(
&
_output_shapes
:  :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:%!

_output_shapes
:	C:!

_output_shapes	
::&	"
 
_output_shapes
:
:!


_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
: :,(
&
_output_shapes
:  :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:%!

_output_shapes
:	C:!

_output_shapes	
::&"
 
_output_shapes
:
:! 

_output_shapes	
::%!!

_output_shapes
:	: "

_output_shapes
::,#(
&
_output_shapes
: :,$(
&
_output_shapes
:  :,%(
&
_output_shapes
: @: &

_output_shapes
:@:,'(
&
_output_shapes
:@@: (

_output_shapes
:@:%)!

_output_shapes
:	C:!*

_output_shapes	
::&+"
 
_output_shapes
:
:!,

_output_shapes	
::%-!

_output_shapes
:	: .

_output_shapes
::/

_output_shapes
: 
§

ø
D__inference_dense_29_layer_call_and_return_conditional_losses_116346

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
¹
H__inference_Edge-to-Node_layer_call_and_return_conditional_losses_115086

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0 
&Edge-to-Node/kernel/Regularizer/SquareSquare=Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @~
%Edge-to-Node/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             §
#Edge-to-Node/kernel/Regularizer/SumSum*Edge-to-Node/kernel/Regularizer/Square:y:0.Edge-to-Node/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%Edge-to-Node/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#Edge-to-Node/kernel/Regularizer/mulMul.Edge-to-Node/kernel/Regularizer/mul/x:output:0,Edge-to-Node/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp6^Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2n
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¿
__inference_loss_fn_0_116404W
=e2e_conv_12_kernel_regularizer_square_readvariableop_resource: 
identity¢4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOpº
4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp=e2e_conv_12_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype0
%e2e_conv_12/kernel/Regularizer/SquareSquare<e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$e2e_conv_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¤
"e2e_conv_12/kernel/Regularizer/SumSum)e2e_conv_12/kernel/Regularizer/Square:y:0-e2e_conv_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$e2e_conv_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¦
"e2e_conv_12/kernel/Regularizer/mulMul-e2e_conv_12/kernel/Regularizer/mul/x:output:0+e2e_conv_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: d
IdentityIdentity&e2e_conv_12/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: }
NoOpNoOp5^e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2l
4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp


ö
D__inference_dense_30_layer_call_and_return_conditional_losses_116393

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
G
+__inference_dropout_49_layer_call_fn_116304

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_49_layer_call_and_return_conditional_losses_115161a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
ã
)__inference_model_16_layer_call_fn_115559
	input_img
input_struc!
unknown: #
	unknown_0:  #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	C
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCall	input_imginput_strucunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_16_layer_call_and_return_conditional_losses_115502o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	input_img:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_struc
Æ
b
F__inference_flatten_16_layer_call_and_return_conditional_losses_116266

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
íW
¨
D__inference_model_16_layer_call_and_return_conditional_losses_115624
	input_img
input_struc,
e2e_conv_12_115563: ,
e2e_conv_13_115566:  -
edge_to_node_115569: @!
edge_to_node_115571:@.
node_to_graph_115574:@@"
node_to_graph_115576:@"
dense_28_115582:	C
dense_28_115584:	#
dense_29_115588:

dense_29_115590:	"
dense_30_115594:	
dense_30_115596:
identity¢$Edge-to-Node/StatefulPartitionedCall¢5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp¢%Node-to-Graph/StatefulPartitionedCall¢6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp¢ dense_28/StatefulPartitionedCall¢ dense_29/StatefulPartitionedCall¢ dense_30/StatefulPartitionedCall¢#e2e_conv_12/StatefulPartitionedCall¢4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp¢#e2e_conv_13/StatefulPartitionedCall¢4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOpñ
#e2e_conv_12/StatefulPartitionedCallStatefulPartitionedCall	input_imge2e_conv_12_115563*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_e2e_conv_12_layer_call_and_return_conditional_losses_115027
#e2e_conv_13/StatefulPartitionedCallStatefulPartitionedCall,e2e_conv_12/StatefulPartitionedCall:output:0e2e_conv_13_115566*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_e2e_conv_13_layer_call_and_return_conditional_losses_115065®
$Edge-to-Node/StatefulPartitionedCallStatefulPartitionedCall,e2e_conv_13/StatefulPartitionedCall:output:0edge_to_node_115569edge_to_node_115571*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Edge-to-Node_layer_call_and_return_conditional_losses_115086³
%Node-to-Graph/StatefulPartitionedCallStatefulPartitionedCall-Edge-to-Node/StatefulPartitionedCall:output:0node_to_graph_115574node_to_graph_115576*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Node-to-Graph_layer_call_and_return_conditional_losses_115109ì
dropout_48/PartitionedCallPartitionedCall.Node-to-Graph/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_48_layer_call_and_return_conditional_losses_115120Ù
flatten_16/PartitionedCallPartitionedCall#dropout_48/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_115128í
concatenate_2/PartitionedCallPartitionedCall#flatten_16/PartitionedCall:output:0input_struc*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_115137
 dense_28/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_28_115582dense_28_115584*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_28_layer_call_and_return_conditional_losses_115150à
dropout_49/PartitionedCallPartitionedCall)dense_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_49_layer_call_and_return_conditional_losses_115161
 dense_29/StatefulPartitionedCallStatefulPartitionedCall#dropout_49/PartitionedCall:output:0dense_29_115588dense_29_115590*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_29_layer_call_and_return_conditional_losses_115174à
dropout_50/PartitionedCallPartitionedCall)dense_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_50_layer_call_and_return_conditional_losses_115185
 dense_30/StatefulPartitionedCallStatefulPartitionedCall#dropout_50/PartitionedCall:output:0dense_30_115594dense_30_115596*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_115198
4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpe2e_conv_12_115563*&
_output_shapes
: *
dtype0
%e2e_conv_12/kernel/Regularizer/SquareSquare<e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$e2e_conv_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¤
"e2e_conv_12/kernel/Regularizer/SumSum)e2e_conv_12/kernel/Regularizer/Square:y:0-e2e_conv_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$e2e_conv_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¦
"e2e_conv_12/kernel/Regularizer/mulMul-e2e_conv_12/kernel/Regularizer/mul/x:output:0+e2e_conv_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpe2e_conv_13_115566*&
_output_shapes
:  *
dtype0
%e2e_conv_13/kernel/Regularizer/SquareSquare<e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  }
$e2e_conv_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¤
"e2e_conv_13/kernel/Regularizer/SumSum)e2e_conv_13/kernel/Regularizer/Square:y:0-e2e_conv_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$e2e_conv_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¦
"e2e_conv_13/kernel/Regularizer/mulMul-e2e_conv_13/kernel/Regularizer/mul/x:output:0+e2e_conv_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOpReadVariableOpedge_to_node_115569*&
_output_shapes
: @*
dtype0 
&Edge-to-Node/kernel/Regularizer/SquareSquare=Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @~
%Edge-to-Node/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             §
#Edge-to-Node/kernel/Regularizer/SumSum*Edge-to-Node/kernel/Regularizer/Square:y:0.Edge-to-Node/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%Edge-to-Node/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#Edge-to-Node/kernel/Regularizer/mulMul.Edge-to-Node/kernel/Regularizer/mul/x:output:0,Edge-to-Node/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOpReadVariableOpnode_to_graph_115574*&
_output_shapes
:@@*
dtype0¢
'Node-to-Graph/kernel/Regularizer/SquareSquare>Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@
&Node-to-Graph/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ª
$Node-to-Graph/kernel/Regularizer/SumSum+Node-to-Graph/kernel/Regularizer/Square:y:0/Node-to-Graph/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&Node-to-Graph/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¬
$Node-to-Graph/kernel/Regularizer/mulMul/Node-to-Graph/kernel/Regularizer/mul/x:output:0-Node-to-Graph/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_30/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
NoOpNoOp%^Edge-to-Node/StatefulPartitionedCall6^Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp&^Node-to-Graph/StatefulPartitionedCall7^Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall$^e2e_conv_12/StatefulPartitionedCall5^e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp$^e2e_conv_13/StatefulPartitionedCall5^e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2L
$Edge-to-Node/StatefulPartitionedCall$Edge-to-Node/StatefulPartitionedCall2n
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp2N
%Node-to-Graph/StatefulPartitionedCall%Node-to-Graph/StatefulPartitionedCall2p
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2J
#e2e_conv_12/StatefulPartitionedCall#e2e_conv_12/StatefulPartitionedCall2l
4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp2J
#e2e_conv_13/StatefulPartitionedCall#e2e_conv_13/StatefulPartitionedCall2l
4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp:Z V
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	input_img:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_struc
¥
G
+__inference_dropout_50_layer_call_fn_116351

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_50_layer_call_and_return_conditional_losses_115185a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£

÷
D__inference_dense_28_layer_call_and_return_conditional_losses_116299

inputs1
matmul_readvariableop_resource:	C.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	C*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿC: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
 
_user_specified_nameinputs

d
+__inference_dropout_48_layer_call_fn_116238

inputs
identity¢StatefulPartitionedCallÉ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_48_layer_call_and_return_conditional_losses_115365w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ü	
e
F__inference_dropout_50_layer_call_and_return_conditional_losses_115286

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»\
	
D__inference_model_16_layer_call_and_return_conditional_losses_115689
	input_img
input_struc,
e2e_conv_12_115628: ,
e2e_conv_13_115631:  -
edge_to_node_115634: @!
edge_to_node_115636:@.
node_to_graph_115639:@@"
node_to_graph_115641:@"
dense_28_115647:	C
dense_28_115649:	#
dense_29_115653:

dense_29_115655:	"
dense_30_115659:	
dense_30_115661:
identity¢$Edge-to-Node/StatefulPartitionedCall¢5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp¢%Node-to-Graph/StatefulPartitionedCall¢6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp¢ dense_28/StatefulPartitionedCall¢ dense_29/StatefulPartitionedCall¢ dense_30/StatefulPartitionedCall¢"dropout_48/StatefulPartitionedCall¢"dropout_49/StatefulPartitionedCall¢"dropout_50/StatefulPartitionedCall¢#e2e_conv_12/StatefulPartitionedCall¢4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp¢#e2e_conv_13/StatefulPartitionedCall¢4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOpñ
#e2e_conv_12/StatefulPartitionedCallStatefulPartitionedCall	input_imge2e_conv_12_115628*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_e2e_conv_12_layer_call_and_return_conditional_losses_115027
#e2e_conv_13/StatefulPartitionedCallStatefulPartitionedCall,e2e_conv_12/StatefulPartitionedCall:output:0e2e_conv_13_115631*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_e2e_conv_13_layer_call_and_return_conditional_losses_115065®
$Edge-to-Node/StatefulPartitionedCallStatefulPartitionedCall,e2e_conv_13/StatefulPartitionedCall:output:0edge_to_node_115634edge_to_node_115636*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Edge-to-Node_layer_call_and_return_conditional_losses_115086³
%Node-to-Graph/StatefulPartitionedCallStatefulPartitionedCall-Edge-to-Node/StatefulPartitionedCall:output:0node_to_graph_115639node_to_graph_115641*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Node-to-Graph_layer_call_and_return_conditional_losses_115109ü
"dropout_48/StatefulPartitionedCallStatefulPartitionedCall.Node-to-Graph/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_48_layer_call_and_return_conditional_losses_115365á
flatten_16/PartitionedCallPartitionedCall+dropout_48/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_115128í
concatenate_2/PartitionedCallPartitionedCall#flatten_16/PartitionedCall:output:0input_struc*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_115137
 dense_28/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_28_115647dense_28_115649*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_28_layer_call_and_return_conditional_losses_115150
"dropout_49/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0#^dropout_48/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_49_layer_call_and_return_conditional_losses_115319
 dense_29/StatefulPartitionedCallStatefulPartitionedCall+dropout_49/StatefulPartitionedCall:output:0dense_29_115653dense_29_115655*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_29_layer_call_and_return_conditional_losses_115174
"dropout_50/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0#^dropout_49/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_50_layer_call_and_return_conditional_losses_115286
 dense_30/StatefulPartitionedCallStatefulPartitionedCall+dropout_50/StatefulPartitionedCall:output:0dense_30_115659dense_30_115661*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_115198
4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpe2e_conv_12_115628*&
_output_shapes
: *
dtype0
%e2e_conv_12/kernel/Regularizer/SquareSquare<e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$e2e_conv_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¤
"e2e_conv_12/kernel/Regularizer/SumSum)e2e_conv_12/kernel/Regularizer/Square:y:0-e2e_conv_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$e2e_conv_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¦
"e2e_conv_12/kernel/Regularizer/mulMul-e2e_conv_12/kernel/Regularizer/mul/x:output:0+e2e_conv_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpe2e_conv_13_115631*&
_output_shapes
:  *
dtype0
%e2e_conv_13/kernel/Regularizer/SquareSquare<e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  }
$e2e_conv_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¤
"e2e_conv_13/kernel/Regularizer/SumSum)e2e_conv_13/kernel/Regularizer/Square:y:0-e2e_conv_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$e2e_conv_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¦
"e2e_conv_13/kernel/Regularizer/mulMul-e2e_conv_13/kernel/Regularizer/mul/x:output:0+e2e_conv_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOpReadVariableOpedge_to_node_115634*&
_output_shapes
: @*
dtype0 
&Edge-to-Node/kernel/Regularizer/SquareSquare=Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @~
%Edge-to-Node/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             §
#Edge-to-Node/kernel/Regularizer/SumSum*Edge-to-Node/kernel/Regularizer/Square:y:0.Edge-to-Node/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%Edge-to-Node/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#Edge-to-Node/kernel/Regularizer/mulMul.Edge-to-Node/kernel/Regularizer/mul/x:output:0,Edge-to-Node/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOpReadVariableOpnode_to_graph_115639*&
_output_shapes
:@@*
dtype0¢
'Node-to-Graph/kernel/Regularizer/SquareSquare>Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@
&Node-to-Graph/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ª
$Node-to-Graph/kernel/Regularizer/SumSum+Node-to-Graph/kernel/Regularizer/Square:y:0/Node-to-Graph/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&Node-to-Graph/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¬
$Node-to-Graph/kernel/Regularizer/mulMul/Node-to-Graph/kernel/Regularizer/mul/x:output:0-Node-to-Graph/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_30/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp%^Edge-to-Node/StatefulPartitionedCall6^Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp&^Node-to-Graph/StatefulPartitionedCall7^Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall#^dropout_48/StatefulPartitionedCall#^dropout_49/StatefulPartitionedCall#^dropout_50/StatefulPartitionedCall$^e2e_conv_12/StatefulPartitionedCall5^e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp$^e2e_conv_13/StatefulPartitionedCall5^e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2L
$Edge-to-Node/StatefulPartitionedCall$Edge-to-Node/StatefulPartitionedCall2n
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp2N
%Node-to-Graph/StatefulPartitionedCall%Node-to-Graph/StatefulPartitionedCall2p
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2H
"dropout_48/StatefulPartitionedCall"dropout_48/StatefulPartitionedCall2H
"dropout_49/StatefulPartitionedCall"dropout_49/StatefulPartitionedCall2H
"dropout_50/StatefulPartitionedCall"dropout_50/StatefulPartitionedCall2J
#e2e_conv_12/StatefulPartitionedCall#e2e_conv_12/StatefulPartitionedCall2l
4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp2J
#e2e_conv_13/StatefulPartitionedCall#e2e_conv_13/StatefulPartitionedCall2l
4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp:Z V
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	input_img:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_struc
ô
£
.__inference_Node-to-Graph_layer_call_fn_116211

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Node-to-Graph_layer_call_and_return_conditional_losses_115109w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ù
d
F__inference_dropout_48_layer_call_and_return_conditional_losses_116243

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
û¸

"__inference__traced_restore_116747
file_prefix=
#assignvariableop_e2e_conv_12_kernel: ?
%assignvariableop_1_e2e_conv_13_kernel:  @
&assignvariableop_2_edge_to_node_kernel: @2
$assignvariableop_3_edge_to_node_bias:@A
'assignvariableop_4_node_to_graph_kernel:@@3
%assignvariableop_5_node_to_graph_bias:@5
"assignvariableop_6_dense_28_kernel:	C/
 assignvariableop_7_dense_28_bias:	6
"assignvariableop_8_dense_29_kernel:
/
 assignvariableop_9_dense_29_bias:	6
#assignvariableop_10_dense_30_kernel:	/
!assignvariableop_11_dense_30_bias:'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: -
assignvariableop_21_accumulator:G
-assignvariableop_22_adam_e2e_conv_12_kernel_m: G
-assignvariableop_23_adam_e2e_conv_13_kernel_m:  H
.assignvariableop_24_adam_edge_to_node_kernel_m: @:
,assignvariableop_25_adam_edge_to_node_bias_m:@I
/assignvariableop_26_adam_node_to_graph_kernel_m:@@;
-assignvariableop_27_adam_node_to_graph_bias_m:@=
*assignvariableop_28_adam_dense_28_kernel_m:	C7
(assignvariableop_29_adam_dense_28_bias_m:	>
*assignvariableop_30_adam_dense_29_kernel_m:
7
(assignvariableop_31_adam_dense_29_bias_m:	=
*assignvariableop_32_adam_dense_30_kernel_m:	6
(assignvariableop_33_adam_dense_30_bias_m:G
-assignvariableop_34_adam_e2e_conv_12_kernel_v: G
-assignvariableop_35_adam_e2e_conv_13_kernel_v:  H
.assignvariableop_36_adam_edge_to_node_kernel_v: @:
,assignvariableop_37_adam_edge_to_node_bias_v:@I
/assignvariableop_38_adam_node_to_graph_kernel_v:@@;
-assignvariableop_39_adam_node_to_graph_bias_v:@=
*assignvariableop_40_adam_dense_28_kernel_v:	C7
(assignvariableop_41_adam_dense_28_bias_v:	>
*assignvariableop_42_adam_dense_29_kernel_v:
7
(assignvariableop_43_adam_dense_29_bias_v:	=
*assignvariableop_44_adam_dense_30_kernel_v:	6
(assignvariableop_45_adam_dense_30_bias_v:
identity_47¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9è
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*
valueB/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÎ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ò
_output_shapes¿
¼:::::::::::::::::::::::::::::::::::::::::::::::*=
dtypes3
12/	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp#assignvariableop_e2e_conv_12_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp%assignvariableop_1_e2e_conv_13_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp&assignvariableop_2_edge_to_node_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp$assignvariableop_3_edge_to_node_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp'assignvariableop_4_node_to_graph_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp%assignvariableop_5_node_to_graph_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_28_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_28_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_29_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_29_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_30_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_30_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_accumulatorIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp-assignvariableop_22_adam_e2e_conv_12_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp-assignvariableop_23_adam_e2e_conv_13_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp.assignvariableop_24_adam_edge_to_node_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp,assignvariableop_25_adam_edge_to_node_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_26AssignVariableOp/assignvariableop_26_adam_node_to_graph_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp-assignvariableop_27_adam_node_to_graph_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_28_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_dense_28_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_dense_29_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_dense_29_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_dense_30_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_dense_30_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp-assignvariableop_34_adam_e2e_conv_12_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp-assignvariableop_35_adam_e2e_conv_13_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp.assignvariableop_36_adam_edge_to_node_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp,assignvariableop_37_adam_edge_to_node_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_38AssignVariableOp/assignvariableop_38_adam_node_to_graph_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp-assignvariableop_39_adam_node_to_graph_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_dense_28_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_dense_28_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_dense_29_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_dense_29_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_dense_30_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_dense_30_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ã
Identity_46Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_47IdentityIdentity_46:output:0^NoOp_1*
T0*
_output_shapes
: °
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_47Identity_47:output:0*q
_input_shapes`
^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ü	
e
F__inference_dropout_49_layer_call_and_return_conditional_losses_115319

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
d
F__inference_dropout_48_layer_call_and_return_conditional_losses_115120

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
´

e
F__inference_dropout_48_layer_call_and_return_conditional_losses_116255

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
µ%
ô
G__inference_e2e_conv_13_layer_call_and_return_conditional_losses_115065

inputs1
readvariableop_resource:  
identity¢ReadVariableOp¢ReadVariableOp_1¢4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOpn
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:  *
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*"
_output_shapes
:  *

begin_mask*
end_mask*
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              s
ReshapeReshapestrided_slice:output:0Reshape/shape:output:0*
T0*&
_output_shapes
:  p
ReadVariableOp_1ReadVariableOpreadvariableop_resource*&
_output_shapes
:  *
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*"
_output_shapes
:  *

begin_mask*
end_mask*
shrink_axis_maskh
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              y
	Reshape_1Reshapestrided_slice_1:output:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:  
convolutionConv2DinputsReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

convolution_1Conv2DinputsReshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :«
concatConcatV2convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0concat/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_1ConcatV2convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0concat_1/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
addAddV2concat:output:0concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:  *
dtype0
%e2e_conv_13/kernel/Regularizer/SquareSquare<e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  }
$e2e_conv_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¤
"e2e_conv_13/kernel/Regularizer/SumSum)e2e_conv_13/kernel/Regularizer/Square:y:0-e2e_conv_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$e2e_conv_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¦
"e2e_conv_13/kernel/Regularizer/mulMul-e2e_conv_13/kernel/Regularizer/mul/x:output:0+e2e_conv_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
NoOpNoOp^ReadVariableOp^ReadVariableOp_15^e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12l
4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
º
s
I__inference_concatenate_2_layer_call_and_return_conditional_losses_115137

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿCW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Z
.__inference_concatenate_2_layer_call_fn_116272
inputs_0
inputs_1
identityÁ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_115137`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1

Þ
$__inference_signature_wrapper_116070
	input_img
input_struc!
unknown: #
	unknown_0:  #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	C
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCall	input_imginput_strucunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_114984o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	input_img:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_struc
ý
»
I__inference_Node-to-Graph_layer_call_and_return_conditional_losses_116228

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0¢
'Node-to-Graph/kernel/Regularizer/SquareSquare>Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@
&Node-to-Graph/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ª
$Node-to-Graph/kernel/Regularizer/SumSum+Node-to-Graph/kernel/Regularizer/Square:y:0/Node-to-Graph/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&Node-to-Graph/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¬
$Node-to-Graph/kernel/Regularizer/mulMul/Node-to-Graph/kernel/Regularizer/mul/x:output:0-Node-to-Graph/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp7^Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2p
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ý
d
F__inference_dropout_49_layer_call_and_return_conditional_losses_115161

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å

)__inference_dense_30_layer_call_fn_116382

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_115198o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò
¢
-__inference_Edge-to-Node_layer_call_fn_116179

inputs!
unknown: @
	unknown_0:@
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Edge-to-Node_layer_call_and_return_conditional_losses_115086w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ý
d
F__inference_dropout_50_layer_call_and_return_conditional_losses_116361

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
d
+__inference_dropout_49_layer_call_fn_116309

inputs
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_49_layer_call_and_return_conditional_losses_115319p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

,__inference_e2e_conv_12_layer_call_fn_116083

inputs!
unknown: 
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_e2e_conv_12_layer_call_and_return_conditional_losses_115027w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´

e
F__inference_dropout_48_layer_call_and_return_conditional_losses_115365

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
³
ã
)__inference_model_16_layer_call_fn_115256
	input_img
input_struc!
unknown: #
	unknown_0:  #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	C
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCall	input_imginput_strucunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_16_layer_call_and_return_conditional_losses_115229o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	input_img:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_struc


ö
D__inference_dense_30_layer_call_and_return_conditional_losses_115198

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
d
F__inference_dropout_50_layer_call_and_return_conditional_losses_115185

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
Ã
__inference_loss_fn_3_116437Y
?node_to_graph_kernel_regularizer_square_readvariableop_resource:@@
identity¢6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp¾
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOpReadVariableOp?node_to_graph_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype0¢
'Node-to-Graph/kernel/Regularizer/SquareSquare>Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@
&Node-to-Graph/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ª
$Node-to-Graph/kernel/Regularizer/SumSum+Node-to-Graph/kernel/Regularizer/Square:y:0/Node-to-Graph/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&Node-to-Graph/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¬
$Node-to-Graph/kernel/Regularizer/mulMul/Node-to-Graph/kernel/Regularizer/mul/x:output:0-Node-to-Graph/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentity(Node-to-Graph/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp7^Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2p
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp
ã°

D__inference_model_16_layer_call_and_return_conditional_losses_116038
inputs_0
inputs_1=
#e2e_conv_12_readvariableop_resource: =
#e2e_conv_13_readvariableop_resource:  E
+edge_to_node_conv2d_readvariableop_resource: @:
,edge_to_node_biasadd_readvariableop_resource:@F
,node_to_graph_conv2d_readvariableop_resource:@@;
-node_to_graph_biasadd_readvariableop_resource:@:
'dense_28_matmul_readvariableop_resource:	C7
(dense_28_biasadd_readvariableop_resource:	;
'dense_29_matmul_readvariableop_resource:
7
(dense_29_biasadd_readvariableop_resource:	:
'dense_30_matmul_readvariableop_resource:	6
(dense_30_biasadd_readvariableop_resource:
identity¢#Edge-to-Node/BiasAdd/ReadVariableOp¢"Edge-to-Node/Conv2D/ReadVariableOp¢5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp¢$Node-to-Graph/BiasAdd/ReadVariableOp¢#Node-to-Graph/Conv2D/ReadVariableOp¢6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp¢dense_28/BiasAdd/ReadVariableOp¢dense_28/MatMul/ReadVariableOp¢dense_29/BiasAdd/ReadVariableOp¢dense_29/MatMul/ReadVariableOp¢dense_30/BiasAdd/ReadVariableOp¢dense_30/MatMul/ReadVariableOp¢e2e_conv_12/ReadVariableOp¢e2e_conv_12/ReadVariableOp_1¢4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp¢e2e_conv_13/ReadVariableOp¢e2e_conv_13/ReadVariableOp_1¢4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp
e2e_conv_12/ReadVariableOpReadVariableOp#e2e_conv_12_readvariableop_resource*&
_output_shapes
: *
dtype0p
e2e_conv_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!e2e_conv_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!e2e_conv_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ã
e2e_conv_12/strided_sliceStridedSlice"e2e_conv_12/ReadVariableOp:value:0(e2e_conv_12/strided_slice/stack:output:0*e2e_conv_12/strided_slice/stack_1:output:0*e2e_conv_12/strided_slice/stack_2:output:0*
Index0*
T0*"
_output_shapes
: *

begin_mask*
end_mask*
shrink_axis_maskr
e2e_conv_12/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             
e2e_conv_12/ReshapeReshape"e2e_conv_12/strided_slice:output:0"e2e_conv_12/Reshape/shape:output:0*
T0*&
_output_shapes
: 
e2e_conv_12/ReadVariableOp_1ReadVariableOp#e2e_conv_12_readvariableop_resource*&
_output_shapes
: *
dtype0r
!e2e_conv_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#e2e_conv_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#e2e_conv_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Í
e2e_conv_12/strided_slice_1StridedSlice$e2e_conv_12/ReadVariableOp_1:value:0*e2e_conv_12/strided_slice_1/stack:output:0,e2e_conv_12/strided_slice_1/stack_1:output:0,e2e_conv_12/strided_slice_1/stack_2:output:0*
Index0*
T0*"
_output_shapes
: *

begin_mask*
end_mask*
shrink_axis_maskt
e2e_conv_12/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             
e2e_conv_12/Reshape_1Reshape$e2e_conv_12/strided_slice_1:output:0$e2e_conv_12/Reshape_1/shape:output:0*
T0*&
_output_shapes
: ¬
e2e_conv_12/convolutionConv2Dinputs_0e2e_conv_12/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
°
e2e_conv_12/convolution_1Conv2Dinputs_0e2e_conv_12/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
Y
e2e_conv_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :£
e2e_conv_12/concatConcatV2"e2e_conv_12/convolution_1:output:0"e2e_conv_12/convolution_1:output:0"e2e_conv_12/convolution_1:output:0"e2e_conv_12/convolution_1:output:0"e2e_conv_12/convolution_1:output:0"e2e_conv_12/convolution_1:output:0"e2e_conv_12/convolution_1:output:0"e2e_conv_12/convolution_1:output:0 e2e_conv_12/concat/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
e2e_conv_12/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
e2e_conv_12/concat_1ConcatV2 e2e_conv_12/convolution:output:0 e2e_conv_12/convolution:output:0 e2e_conv_12/convolution:output:0 e2e_conv_12/convolution:output:0 e2e_conv_12/convolution:output:0 e2e_conv_12/convolution:output:0 e2e_conv_12/convolution:output:0 e2e_conv_12/convolution:output:0"e2e_conv_12/concat_1/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
e2e_conv_12/addAddV2e2e_conv_12/concat:output:0e2e_conv_12/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
e2e_conv_13/ReadVariableOpReadVariableOp#e2e_conv_13_readvariableop_resource*&
_output_shapes
:  *
dtype0p
e2e_conv_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!e2e_conv_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!e2e_conv_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ã
e2e_conv_13/strided_sliceStridedSlice"e2e_conv_13/ReadVariableOp:value:0(e2e_conv_13/strided_slice/stack:output:0*e2e_conv_13/strided_slice/stack_1:output:0*e2e_conv_13/strided_slice/stack_2:output:0*
Index0*
T0*"
_output_shapes
:  *

begin_mask*
end_mask*
shrink_axis_maskr
e2e_conv_13/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              
e2e_conv_13/ReshapeReshape"e2e_conv_13/strided_slice:output:0"e2e_conv_13/Reshape/shape:output:0*
T0*&
_output_shapes
:  
e2e_conv_13/ReadVariableOp_1ReadVariableOp#e2e_conv_13_readvariableop_resource*&
_output_shapes
:  *
dtype0r
!e2e_conv_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#e2e_conv_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#e2e_conv_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Í
e2e_conv_13/strided_slice_1StridedSlice$e2e_conv_13/ReadVariableOp_1:value:0*e2e_conv_13/strided_slice_1/stack:output:0,e2e_conv_13/strided_slice_1/stack_1:output:0,e2e_conv_13/strided_slice_1/stack_2:output:0*
Index0*
T0*"
_output_shapes
:  *

begin_mask*
end_mask*
shrink_axis_maskt
e2e_conv_13/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              
e2e_conv_13/Reshape_1Reshape$e2e_conv_13/strided_slice_1:output:0$e2e_conv_13/Reshape_1/shape:output:0*
T0*&
_output_shapes
:  ·
e2e_conv_13/convolutionConv2De2e_conv_12/add:z:0e2e_conv_13/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
»
e2e_conv_13/convolution_1Conv2De2e_conv_12/add:z:0e2e_conv_13/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
Y
e2e_conv_13/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :£
e2e_conv_13/concatConcatV2"e2e_conv_13/convolution_1:output:0"e2e_conv_13/convolution_1:output:0"e2e_conv_13/convolution_1:output:0"e2e_conv_13/convolution_1:output:0"e2e_conv_13/convolution_1:output:0"e2e_conv_13/convolution_1:output:0"e2e_conv_13/convolution_1:output:0"e2e_conv_13/convolution_1:output:0 e2e_conv_13/concat/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
e2e_conv_13/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
e2e_conv_13/concat_1ConcatV2 e2e_conv_13/convolution:output:0 e2e_conv_13/convolution:output:0 e2e_conv_13/convolution:output:0 e2e_conv_13/convolution:output:0 e2e_conv_13/convolution:output:0 e2e_conv_13/convolution:output:0 e2e_conv_13/convolution:output:0 e2e_conv_13/convolution:output:0"e2e_conv_13/concat_1/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
e2e_conv_13/addAddV2e2e_conv_13/concat:output:0e2e_conv_13/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"Edge-to-Node/Conv2D/ReadVariableOpReadVariableOp+edge_to_node_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Á
Edge-to-Node/Conv2DConv2De2e_conv_13/add:z:0*Edge-to-Node/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

#Edge-to-Node/BiasAdd/ReadVariableOpReadVariableOp,edge_to_node_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¤
Edge-to-Node/BiasAddBiasAddEdge-to-Node/Conv2D:output:0+Edge-to-Node/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
Edge-to-Node/ReluReluEdge-to-Node/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
#Node-to-Graph/Conv2D/ReadVariableOpReadVariableOp,node_to_graph_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ï
Node-to-Graph/Conv2DConv2DEdge-to-Node/Relu:activations:0+Node-to-Graph/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

$Node-to-Graph/BiasAdd/ReadVariableOpReadVariableOp-node_to_graph_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0§
Node-to-Graph/BiasAddBiasAddNode-to-Graph/Conv2D:output:0,Node-to-Graph/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
Node-to-Graph/ReluReluNode-to-Graph/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
dropout_48/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_48/dropout/MulMul Node-to-Graph/Relu:activations:0!dropout_48/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
dropout_48/dropout/ShapeShape Node-to-Graph/Relu:activations:0*
T0*
_output_shapes
:ª
/dropout_48/dropout/random_uniform/RandomUniformRandomUniform!dropout_48/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0f
!dropout_48/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ï
dropout_48/dropout/GreaterEqualGreaterEqual8dropout_48/dropout/random_uniform/RandomUniform:output:0*dropout_48/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout_48/dropout/CastCast#dropout_48/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout_48/dropout/Mul_1Muldropout_48/dropout/Mul:z:0dropout_48/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
flatten_16/ReshapeReshapedropout_48/dropout/Mul_1:z:0flatten_16/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¦
concatenate_2/concatConcatV2flatten_16/Reshape:output:0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes
:	C*
dtype0
dense_28/MatMulMatMulconcatenate_2/concat:output:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
dropout_49/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_49/dropout/MulMuldense_28/Relu:activations:0!dropout_49/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dropout_49/dropout/ShapeShapedense_28/Relu:activations:0*
T0*
_output_shapes
:£
/dropout_49/dropout/random_uniform/RandomUniformRandomUniform!dropout_49/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0f
!dropout_49/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?È
dropout_49/dropout/GreaterEqualGreaterEqual8dropout_49/dropout/random_uniform/RandomUniform:output:0*dropout_49/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_49/dropout/CastCast#dropout_49/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_49/dropout/Mul_1Muldropout_49/dropout/Mul:z:0dropout_49/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_29/MatMulMatMuldropout_49/dropout/Mul_1:z:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_29/ReluReludense_29/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
dropout_50/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_50/dropout/MulMuldense_29/Relu:activations:0!dropout_50/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dropout_50/dropout/ShapeShapedense_29/Relu:activations:0*
T0*
_output_shapes
:£
/dropout_50/dropout/random_uniform/RandomUniformRandomUniform!dropout_50/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0f
!dropout_50/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?È
dropout_50/dropout/GreaterEqualGreaterEqual8dropout_50/dropout/random_uniform/RandomUniform:output:0*dropout_50/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_50/dropout/CastCast#dropout_50/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_50/dropout/Mul_1Muldropout_50/dropout/Mul:z:0dropout_50/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_30/MatMulMatMuldropout_50/dropout/Mul_1:z:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_30/SigmoidSigmoiddense_30/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp#e2e_conv_12_readvariableop_resource*&
_output_shapes
: *
dtype0
%e2e_conv_12/kernel/Regularizer/SquareSquare<e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$e2e_conv_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¤
"e2e_conv_12/kernel/Regularizer/SumSum)e2e_conv_12/kernel/Regularizer/Square:y:0-e2e_conv_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$e2e_conv_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¦
"e2e_conv_12/kernel/Regularizer/mulMul-e2e_conv_12/kernel/Regularizer/mul/x:output:0+e2e_conv_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
:  
4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp#e2e_conv_13_readvariableop_resource*&
_output_shapes
:  *
dtype0
%e2e_conv_13/kernel/Regularizer/SquareSquare<e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  }
$e2e_conv_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¤
"e2e_conv_13/kernel/Regularizer/SumSum)e2e_conv_13/kernel/Regularizer/Square:y:0-e2e_conv_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$e2e_conv_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¦
"e2e_conv_13/kernel/Regularizer/mulMul-e2e_conv_13/kernel/Regularizer/mul/x:output:0+e2e_conv_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ©
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+edge_to_node_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0 
&Edge-to-Node/kernel/Regularizer/SquareSquare=Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @~
%Edge-to-Node/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             §
#Edge-to-Node/kernel/Regularizer/SumSum*Edge-to-Node/kernel/Regularizer/Square:y:0.Edge-to-Node/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%Edge-to-Node/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#Edge-to-Node/kernel/Regularizer/mulMul.Edge-to-Node/kernel/Regularizer/mul/x:output:0,Edge-to-Node/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: «
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOpReadVariableOp,node_to_graph_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0¢
'Node-to-Graph/kernel/Regularizer/SquareSquare>Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@
&Node-to-Graph/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ª
$Node-to-Graph/kernel/Regularizer/SumSum+Node-to-Graph/kernel/Regularizer/Square:y:0/Node-to-Graph/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&Node-to-Graph/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¬
$Node-to-Graph/kernel/Regularizer/mulMul/Node-to-Graph/kernel/Regularizer/mul/x:output:0-Node-to-Graph/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
IdentityIdentitydense_30/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ
NoOpNoOp$^Edge-to-Node/BiasAdd/ReadVariableOp#^Edge-to-Node/Conv2D/ReadVariableOp6^Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp%^Node-to-Graph/BiasAdd/ReadVariableOp$^Node-to-Graph/Conv2D/ReadVariableOp7^Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp^e2e_conv_12/ReadVariableOp^e2e_conv_12/ReadVariableOp_15^e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp^e2e_conv_13/ReadVariableOp^e2e_conv_13/ReadVariableOp_15^e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2J
#Edge-to-Node/BiasAdd/ReadVariableOp#Edge-to-Node/BiasAdd/ReadVariableOp2H
"Edge-to-Node/Conv2D/ReadVariableOp"Edge-to-Node/Conv2D/ReadVariableOp2n
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp2L
$Node-to-Graph/BiasAdd/ReadVariableOp$Node-to-Graph/BiasAdd/ReadVariableOp2J
#Node-to-Graph/Conv2D/ReadVariableOp#Node-to-Graph/Conv2D/ReadVariableOp2p
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp28
e2e_conv_12/ReadVariableOpe2e_conv_12/ReadVariableOp2<
e2e_conv_12/ReadVariableOp_1e2e_conv_12/ReadVariableOp_12l
4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp28
e2e_conv_13/ReadVariableOpe2e_conv_13/ReadVariableOp2<
e2e_conv_13/ReadVariableOp_1e2e_conv_13/ReadVariableOp_12l
4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
µ%
ô
G__inference_e2e_conv_12_layer_call_and_return_conditional_losses_116117

inputs1
readvariableop_resource: 
identity¢ReadVariableOp¢ReadVariableOp_1¢4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOpn
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
: *
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*"
_output_shapes
: *

begin_mask*
end_mask*
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             s
ReshapeReshapestrided_slice:output:0Reshape/shape:output:0*
T0*&
_output_shapes
: p
ReadVariableOp_1ReadVariableOpreadvariableop_resource*&
_output_shapes
: *
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*"
_output_shapes
: *

begin_mask*
end_mask*
shrink_axis_maskh
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             y
	Reshape_1Reshapestrided_slice_1:output:0Reshape_1/shape:output:0*
T0*&
_output_shapes
: 
convolutionConv2DinputsReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

convolution_1Conv2DinputsReshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :«
concatConcatV2convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0concat/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_1ConcatV2convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0concat_1/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
addAddV2concat:output:0concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
: *
dtype0
%e2e_conv_12/kernel/Regularizer/SquareSquare<e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$e2e_conv_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¤
"e2e_conv_12/kernel/Regularizer/SumSum)e2e_conv_12/kernel/Regularizer/Square:y:0-e2e_conv_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$e2e_conv_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¦
"e2e_conv_12/kernel/Regularizer/mulMul-e2e_conv_12/kernel/Regularizer/mul/x:output:0+e2e_conv_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¡
NoOpNoOp^ReadVariableOp^ReadVariableOp_15^e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12l
4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

,__inference_e2e_conv_13_layer_call_fn_116130

inputs!
unknown:  
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_e2e_conv_13_layer_call_and_return_conditional_losses_115065w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
§
ß
)__inference_model_16_layer_call_fn_115749
inputs_0
inputs_1!
unknown: #
	unknown_0:  #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	C
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:	

unknown_10:
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_16_layer_call_and_return_conditional_losses_115229o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Æ
b
F__inference_flatten_16_layer_call_and_return_conditional_losses_115128

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Â
u
I__inference_concatenate_2_layer_call_and_return_conditional_losses_116279
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿCW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
É

)__inference_dense_29_layer_call_fn_116335

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_29_layer_call_and_return_conditional_losses_115174p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÙW
¢
D__inference_model_16_layer_call_and_return_conditional_losses_115229

inputs
inputs_1,
e2e_conv_12_115028: ,
e2e_conv_13_115066:  -
edge_to_node_115087: @!
edge_to_node_115089:@.
node_to_graph_115110:@@"
node_to_graph_115112:@"
dense_28_115151:	C
dense_28_115153:	#
dense_29_115175:

dense_29_115177:	"
dense_30_115199:	
dense_30_115201:
identity¢$Edge-to-Node/StatefulPartitionedCall¢5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp¢%Node-to-Graph/StatefulPartitionedCall¢6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp¢ dense_28/StatefulPartitionedCall¢ dense_29/StatefulPartitionedCall¢ dense_30/StatefulPartitionedCall¢#e2e_conv_12/StatefulPartitionedCall¢4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp¢#e2e_conv_13/StatefulPartitionedCall¢4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOpî
#e2e_conv_12/StatefulPartitionedCallStatefulPartitionedCallinputse2e_conv_12_115028*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_e2e_conv_12_layer_call_and_return_conditional_losses_115027
#e2e_conv_13/StatefulPartitionedCallStatefulPartitionedCall,e2e_conv_12/StatefulPartitionedCall:output:0e2e_conv_13_115066*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_e2e_conv_13_layer_call_and_return_conditional_losses_115065®
$Edge-to-Node/StatefulPartitionedCallStatefulPartitionedCall,e2e_conv_13/StatefulPartitionedCall:output:0edge_to_node_115087edge_to_node_115089*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Edge-to-Node_layer_call_and_return_conditional_losses_115086³
%Node-to-Graph/StatefulPartitionedCallStatefulPartitionedCall-Edge-to-Node/StatefulPartitionedCall:output:0node_to_graph_115110node_to_graph_115112*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Node-to-Graph_layer_call_and_return_conditional_losses_115109ì
dropout_48/PartitionedCallPartitionedCall.Node-to-Graph/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_48_layer_call_and_return_conditional_losses_115120Ù
flatten_16/PartitionedCallPartitionedCall#dropout_48/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_115128ê
concatenate_2/PartitionedCallPartitionedCall#flatten_16/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_115137
 dense_28/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_28_115151dense_28_115153*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_28_layer_call_and_return_conditional_losses_115150à
dropout_49/PartitionedCallPartitionedCall)dense_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_49_layer_call_and_return_conditional_losses_115161
 dense_29/StatefulPartitionedCallStatefulPartitionedCall#dropout_49/PartitionedCall:output:0dense_29_115175dense_29_115177*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_29_layer_call_and_return_conditional_losses_115174à
dropout_50/PartitionedCallPartitionedCall)dense_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_50_layer_call_and_return_conditional_losses_115185
 dense_30/StatefulPartitionedCallStatefulPartitionedCall#dropout_50/PartitionedCall:output:0dense_30_115199dense_30_115201*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_115198
4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpe2e_conv_12_115028*&
_output_shapes
: *
dtype0
%e2e_conv_12/kernel/Regularizer/SquareSquare<e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$e2e_conv_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¤
"e2e_conv_12/kernel/Regularizer/SumSum)e2e_conv_12/kernel/Regularizer/Square:y:0-e2e_conv_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$e2e_conv_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¦
"e2e_conv_12/kernel/Regularizer/mulMul-e2e_conv_12/kernel/Regularizer/mul/x:output:0+e2e_conv_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpe2e_conv_13_115066*&
_output_shapes
:  *
dtype0
%e2e_conv_13/kernel/Regularizer/SquareSquare<e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  }
$e2e_conv_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¤
"e2e_conv_13/kernel/Regularizer/SumSum)e2e_conv_13/kernel/Regularizer/Square:y:0-e2e_conv_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$e2e_conv_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¦
"e2e_conv_13/kernel/Regularizer/mulMul-e2e_conv_13/kernel/Regularizer/mul/x:output:0+e2e_conv_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOpReadVariableOpedge_to_node_115087*&
_output_shapes
: @*
dtype0 
&Edge-to-Node/kernel/Regularizer/SquareSquare=Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @~
%Edge-to-Node/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             §
#Edge-to-Node/kernel/Regularizer/SumSum*Edge-to-Node/kernel/Regularizer/Square:y:0.Edge-to-Node/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%Edge-to-Node/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#Edge-to-Node/kernel/Regularizer/mulMul.Edge-to-Node/kernel/Regularizer/mul/x:output:0,Edge-to-Node/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOpReadVariableOpnode_to_graph_115110*&
_output_shapes
:@@*
dtype0¢
'Node-to-Graph/kernel/Regularizer/SquareSquare>Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@
&Node-to-Graph/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ª
$Node-to-Graph/kernel/Regularizer/SumSum+Node-to-Graph/kernel/Regularizer/Square:y:0/Node-to-Graph/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&Node-to-Graph/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¬
$Node-to-Graph/kernel/Regularizer/mulMul/Node-to-Graph/kernel/Regularizer/mul/x:output:0-Node-to-Graph/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_30/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
NoOpNoOp%^Edge-to-Node/StatefulPartitionedCall6^Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp&^Node-to-Graph/StatefulPartitionedCall7^Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall$^e2e_conv_12/StatefulPartitionedCall5^e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp$^e2e_conv_13/StatefulPartitionedCall5^e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2L
$Edge-to-Node/StatefulPartitionedCall$Edge-to-Node/StatefulPartitionedCall2n
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp2N
%Node-to-Graph/StatefulPartitionedCall%Node-to-Graph/StatefulPartitionedCall2p
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2J
#e2e_conv_12/StatefulPartitionedCall#e2e_conv_12/StatefulPartitionedCall2l
4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp2J
#e2e_conv_13/StatefulPartitionedCall#e2e_conv_13/StatefulPartitionedCall2l
4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü	
e
F__inference_dropout_50_layer_call_and_return_conditional_losses_116373

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
¹
H__inference_Edge-to-Node_layer_call_and_return_conditional_losses_116196

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0 
&Edge-to-Node/kernel/Regularizer/SquareSquare=Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @~
%Edge-to-Node/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             §
#Edge-to-Node/kernel/Regularizer/SumSum*Edge-to-Node/kernel/Regularizer/Square:y:0.Edge-to-Node/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%Edge-to-Node/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#Edge-to-Node/kernel/Regularizer/mulMul.Edge-to-Node/kernel/Regularizer/mul/x:output:0,Edge-to-Node/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp6^Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2n
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

Á
__inference_loss_fn_2_116426X
>edge_to_node_kernel_regularizer_square_readvariableop_resource: @
identity¢5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp¼
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>edge_to_node_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype0 
&Edge-to-Node/kernel/Regularizer/SquareSquare=Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @~
%Edge-to-Node/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             §
#Edge-to-Node/kernel/Regularizer/SumSum*Edge-to-Node/kernel/Regularizer/Square:y:0.Edge-to-Node/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%Edge-to-Node/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#Edge-to-Node/kernel/Regularizer/mulMul.Edge-to-Node/kernel/Regularizer/mul/x:output:0,Edge-to-Node/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentity'Edge-to-Node/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp
¤
ü
!__inference__wrapped_model_114984
	input_img
input_strucF
,model_16_e2e_conv_12_readvariableop_resource: F
,model_16_e2e_conv_13_readvariableop_resource:  N
4model_16_edge_to_node_conv2d_readvariableop_resource: @C
5model_16_edge_to_node_biasadd_readvariableop_resource:@O
5model_16_node_to_graph_conv2d_readvariableop_resource:@@D
6model_16_node_to_graph_biasadd_readvariableop_resource:@C
0model_16_dense_28_matmul_readvariableop_resource:	C@
1model_16_dense_28_biasadd_readvariableop_resource:	D
0model_16_dense_29_matmul_readvariableop_resource:
@
1model_16_dense_29_biasadd_readvariableop_resource:	C
0model_16_dense_30_matmul_readvariableop_resource:	?
1model_16_dense_30_biasadd_readvariableop_resource:
identity¢,model_16/Edge-to-Node/BiasAdd/ReadVariableOp¢+model_16/Edge-to-Node/Conv2D/ReadVariableOp¢-model_16/Node-to-Graph/BiasAdd/ReadVariableOp¢,model_16/Node-to-Graph/Conv2D/ReadVariableOp¢(model_16/dense_28/BiasAdd/ReadVariableOp¢'model_16/dense_28/MatMul/ReadVariableOp¢(model_16/dense_29/BiasAdd/ReadVariableOp¢'model_16/dense_29/MatMul/ReadVariableOp¢(model_16/dense_30/BiasAdd/ReadVariableOp¢'model_16/dense_30/MatMul/ReadVariableOp¢#model_16/e2e_conv_12/ReadVariableOp¢%model_16/e2e_conv_12/ReadVariableOp_1¢#model_16/e2e_conv_13/ReadVariableOp¢%model_16/e2e_conv_13/ReadVariableOp_1
#model_16/e2e_conv_12/ReadVariableOpReadVariableOp,model_16_e2e_conv_12_readvariableop_resource*&
_output_shapes
: *
dtype0y
(model_16/e2e_conv_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*model_16/e2e_conv_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*model_16/e2e_conv_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ð
"model_16/e2e_conv_12/strided_sliceStridedSlice+model_16/e2e_conv_12/ReadVariableOp:value:01model_16/e2e_conv_12/strided_slice/stack:output:03model_16/e2e_conv_12/strided_slice/stack_1:output:03model_16/e2e_conv_12/strided_slice/stack_2:output:0*
Index0*
T0*"
_output_shapes
: *

begin_mask*
end_mask*
shrink_axis_mask{
"model_16/e2e_conv_12/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             ²
model_16/e2e_conv_12/ReshapeReshape+model_16/e2e_conv_12/strided_slice:output:0+model_16/e2e_conv_12/Reshape/shape:output:0*
T0*&
_output_shapes
: 
%model_16/e2e_conv_12/ReadVariableOp_1ReadVariableOp,model_16_e2e_conv_12_readvariableop_resource*&
_output_shapes
: *
dtype0{
*model_16/e2e_conv_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,model_16/e2e_conv_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,model_16/e2e_conv_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ú
$model_16/e2e_conv_12/strided_slice_1StridedSlice-model_16/e2e_conv_12/ReadVariableOp_1:value:03model_16/e2e_conv_12/strided_slice_1/stack:output:05model_16/e2e_conv_12/strided_slice_1/stack_1:output:05model_16/e2e_conv_12/strided_slice_1/stack_2:output:0*
Index0*
T0*"
_output_shapes
: *

begin_mask*
end_mask*
shrink_axis_mask}
$model_16/e2e_conv_12/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             ¸
model_16/e2e_conv_12/Reshape_1Reshape-model_16/e2e_conv_12/strided_slice_1:output:0-model_16/e2e_conv_12/Reshape_1/shape:output:0*
T0*&
_output_shapes
: ¿
 model_16/e2e_conv_12/convolutionConv2D	input_img%model_16/e2e_conv_12/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
Ã
"model_16/e2e_conv_12/convolution_1Conv2D	input_img'model_16/e2e_conv_12/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
b
 model_16/e2e_conv_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ý
model_16/e2e_conv_12/concatConcatV2+model_16/e2e_conv_12/convolution_1:output:0+model_16/e2e_conv_12/convolution_1:output:0+model_16/e2e_conv_12/convolution_1:output:0+model_16/e2e_conv_12/convolution_1:output:0+model_16/e2e_conv_12/convolution_1:output:0+model_16/e2e_conv_12/convolution_1:output:0+model_16/e2e_conv_12/convolution_1:output:0+model_16/e2e_conv_12/convolution_1:output:0)model_16/e2e_conv_12/concat/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
"model_16/e2e_conv_12/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :ñ
model_16/e2e_conv_12/concat_1ConcatV2)model_16/e2e_conv_12/convolution:output:0)model_16/e2e_conv_12/convolution:output:0)model_16/e2e_conv_12/convolution:output:0)model_16/e2e_conv_12/convolution:output:0)model_16/e2e_conv_12/convolution:output:0)model_16/e2e_conv_12/convolution:output:0)model_16/e2e_conv_12/convolution:output:0)model_16/e2e_conv_12/convolution:output:0+model_16/e2e_conv_12/concat_1/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©
model_16/e2e_conv_12/addAddV2$model_16/e2e_conv_12/concat:output:0&model_16/e2e_conv_12/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
#model_16/e2e_conv_13/ReadVariableOpReadVariableOp,model_16_e2e_conv_13_readvariableop_resource*&
_output_shapes
:  *
dtype0y
(model_16/e2e_conv_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*model_16/e2e_conv_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*model_16/e2e_conv_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ð
"model_16/e2e_conv_13/strided_sliceStridedSlice+model_16/e2e_conv_13/ReadVariableOp:value:01model_16/e2e_conv_13/strided_slice/stack:output:03model_16/e2e_conv_13/strided_slice/stack_1:output:03model_16/e2e_conv_13/strided_slice/stack_2:output:0*
Index0*
T0*"
_output_shapes
:  *

begin_mask*
end_mask*
shrink_axis_mask{
"model_16/e2e_conv_13/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              ²
model_16/e2e_conv_13/ReshapeReshape+model_16/e2e_conv_13/strided_slice:output:0+model_16/e2e_conv_13/Reshape/shape:output:0*
T0*&
_output_shapes
:  
%model_16/e2e_conv_13/ReadVariableOp_1ReadVariableOp,model_16_e2e_conv_13_readvariableop_resource*&
_output_shapes
:  *
dtype0{
*model_16/e2e_conv_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,model_16/e2e_conv_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,model_16/e2e_conv_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ú
$model_16/e2e_conv_13/strided_slice_1StridedSlice-model_16/e2e_conv_13/ReadVariableOp_1:value:03model_16/e2e_conv_13/strided_slice_1/stack:output:05model_16/e2e_conv_13/strided_slice_1/stack_1:output:05model_16/e2e_conv_13/strided_slice_1/stack_2:output:0*
Index0*
T0*"
_output_shapes
:  *

begin_mask*
end_mask*
shrink_axis_mask}
$model_16/e2e_conv_13/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              ¸
model_16/e2e_conv_13/Reshape_1Reshape-model_16/e2e_conv_13/strided_slice_1:output:0-model_16/e2e_conv_13/Reshape_1/shape:output:0*
T0*&
_output_shapes
:  Ò
 model_16/e2e_conv_13/convolutionConv2Dmodel_16/e2e_conv_12/add:z:0%model_16/e2e_conv_13/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
Ö
"model_16/e2e_conv_13/convolution_1Conv2Dmodel_16/e2e_conv_12/add:z:0'model_16/e2e_conv_13/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
b
 model_16/e2e_conv_13/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ý
model_16/e2e_conv_13/concatConcatV2+model_16/e2e_conv_13/convolution_1:output:0+model_16/e2e_conv_13/convolution_1:output:0+model_16/e2e_conv_13/convolution_1:output:0+model_16/e2e_conv_13/convolution_1:output:0+model_16/e2e_conv_13/convolution_1:output:0+model_16/e2e_conv_13/convolution_1:output:0+model_16/e2e_conv_13/convolution_1:output:0+model_16/e2e_conv_13/convolution_1:output:0)model_16/e2e_conv_13/concat/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
"model_16/e2e_conv_13/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :ñ
model_16/e2e_conv_13/concat_1ConcatV2)model_16/e2e_conv_13/convolution:output:0)model_16/e2e_conv_13/convolution:output:0)model_16/e2e_conv_13/convolution:output:0)model_16/e2e_conv_13/convolution:output:0)model_16/e2e_conv_13/convolution:output:0)model_16/e2e_conv_13/convolution:output:0)model_16/e2e_conv_13/convolution:output:0)model_16/e2e_conv_13/convolution:output:0+model_16/e2e_conv_13/concat_1/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©
model_16/e2e_conv_13/addAddV2$model_16/e2e_conv_13/concat:output:0&model_16/e2e_conv_13/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¨
+model_16/Edge-to-Node/Conv2D/ReadVariableOpReadVariableOp4model_16_edge_to_node_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ü
model_16/Edge-to-Node/Conv2DConv2Dmodel_16/e2e_conv_13/add:z:03model_16/Edge-to-Node/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

,model_16/Edge-to-Node/BiasAdd/ReadVariableOpReadVariableOp5model_16_edge_to_node_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¿
model_16/Edge-to-Node/BiasAddBiasAdd%model_16/Edge-to-Node/Conv2D:output:04model_16/Edge-to-Node/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
model_16/Edge-to-Node/ReluRelu&model_16/Edge-to-Node/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ª
,model_16/Node-to-Graph/Conv2D/ReadVariableOpReadVariableOp5model_16_node_to_graph_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ê
model_16/Node-to-Graph/Conv2DConv2D(model_16/Edge-to-Node/Relu:activations:04model_16/Node-to-Graph/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
 
-model_16/Node-to-Graph/BiasAdd/ReadVariableOpReadVariableOp6model_16_node_to_graph_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Â
model_16/Node-to-Graph/BiasAddBiasAdd&model_16/Node-to-Graph/Conv2D:output:05model_16/Node-to-Graph/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
model_16/Node-to-Graph/ReluRelu'model_16/Node-to-Graph/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
model_16/dropout_48/IdentityIdentity)model_16/Node-to-Graph/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
model_16/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   £
model_16/flatten_16/ReshapeReshape%model_16/dropout_48/Identity:output:0"model_16/flatten_16/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
"model_16/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ä
model_16/concatenate_2/concatConcatV2$model_16/flatten_16/Reshape:output:0input_struc+model_16/concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
'model_16/dense_28/MatMul/ReadVariableOpReadVariableOp0model_16_dense_28_matmul_readvariableop_resource*
_output_shapes
:	C*
dtype0®
model_16/dense_28/MatMulMatMul&model_16/concatenate_2/concat:output:0/model_16/dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(model_16/dense_28/BiasAdd/ReadVariableOpReadVariableOp1model_16_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
model_16/dense_28/BiasAddBiasAdd"model_16/dense_28/MatMul:product:00model_16/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
model_16/dense_28/ReluRelu"model_16/dense_28/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_16/dropout_49/IdentityIdentity$model_16/dense_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_16/dense_29/MatMul/ReadVariableOpReadVariableOp0model_16_dense_29_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0­
model_16/dense_29/MatMulMatMul%model_16/dropout_49/Identity:output:0/model_16/dense_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(model_16/dense_29/BiasAdd/ReadVariableOpReadVariableOp1model_16_dense_29_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
model_16/dense_29/BiasAddBiasAdd"model_16/dense_29/MatMul:product:00model_16/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
model_16/dense_29/ReluRelu"model_16/dense_29/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_16/dropout_50/IdentityIdentity$model_16/dense_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_16/dense_30/MatMul/ReadVariableOpReadVariableOp0model_16_dense_30_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¬
model_16/dense_30/MatMulMatMul%model_16/dropout_50/Identity:output:0/model_16/dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(model_16/dense_30/BiasAdd/ReadVariableOpReadVariableOp1model_16_dense_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¬
model_16/dense_30/BiasAddBiasAdd"model_16/dense_30/MatMul:product:00model_16/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
model_16/dense_30/SigmoidSigmoid"model_16/dense_30/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
IdentityIdentitymodel_16/dense_30/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp-^model_16/Edge-to-Node/BiasAdd/ReadVariableOp,^model_16/Edge-to-Node/Conv2D/ReadVariableOp.^model_16/Node-to-Graph/BiasAdd/ReadVariableOp-^model_16/Node-to-Graph/Conv2D/ReadVariableOp)^model_16/dense_28/BiasAdd/ReadVariableOp(^model_16/dense_28/MatMul/ReadVariableOp)^model_16/dense_29/BiasAdd/ReadVariableOp(^model_16/dense_29/MatMul/ReadVariableOp)^model_16/dense_30/BiasAdd/ReadVariableOp(^model_16/dense_30/MatMul/ReadVariableOp$^model_16/e2e_conv_12/ReadVariableOp&^model_16/e2e_conv_12/ReadVariableOp_1$^model_16/e2e_conv_13/ReadVariableOp&^model_16/e2e_conv_13/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2\
,model_16/Edge-to-Node/BiasAdd/ReadVariableOp,model_16/Edge-to-Node/BiasAdd/ReadVariableOp2Z
+model_16/Edge-to-Node/Conv2D/ReadVariableOp+model_16/Edge-to-Node/Conv2D/ReadVariableOp2^
-model_16/Node-to-Graph/BiasAdd/ReadVariableOp-model_16/Node-to-Graph/BiasAdd/ReadVariableOp2\
,model_16/Node-to-Graph/Conv2D/ReadVariableOp,model_16/Node-to-Graph/Conv2D/ReadVariableOp2T
(model_16/dense_28/BiasAdd/ReadVariableOp(model_16/dense_28/BiasAdd/ReadVariableOp2R
'model_16/dense_28/MatMul/ReadVariableOp'model_16/dense_28/MatMul/ReadVariableOp2T
(model_16/dense_29/BiasAdd/ReadVariableOp(model_16/dense_29/BiasAdd/ReadVariableOp2R
'model_16/dense_29/MatMul/ReadVariableOp'model_16/dense_29/MatMul/ReadVariableOp2T
(model_16/dense_30/BiasAdd/ReadVariableOp(model_16/dense_30/BiasAdd/ReadVariableOp2R
'model_16/dense_30/MatMul/ReadVariableOp'model_16/dense_30/MatMul/ReadVariableOp2J
#model_16/e2e_conv_12/ReadVariableOp#model_16/e2e_conv_12/ReadVariableOp2N
%model_16/e2e_conv_12/ReadVariableOp_1%model_16/e2e_conv_12/ReadVariableOp_12J
#model_16/e2e_conv_13/ReadVariableOp#model_16/e2e_conv_13/ReadVariableOp2N
%model_16/e2e_conv_13/ReadVariableOp_1%model_16/e2e_conv_13/ReadVariableOp_1:Z V
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	input_img:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_struc
£

÷
D__inference_dense_28_layer_call_and_return_conditional_losses_115150

inputs1
matmul_readvariableop_resource:	C.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	C*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿC: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
 
_user_specified_nameinputs
±
G
+__inference_flatten_16_layer_call_fn_116260

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_115128`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
§\
	
D__inference_model_16_layer_call_and_return_conditional_losses_115502

inputs
inputs_1,
e2e_conv_12_115441: ,
e2e_conv_13_115444:  -
edge_to_node_115447: @!
edge_to_node_115449:@.
node_to_graph_115452:@@"
node_to_graph_115454:@"
dense_28_115460:	C
dense_28_115462:	#
dense_29_115466:

dense_29_115468:	"
dense_30_115472:	
dense_30_115474:
identity¢$Edge-to-Node/StatefulPartitionedCall¢5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp¢%Node-to-Graph/StatefulPartitionedCall¢6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp¢ dense_28/StatefulPartitionedCall¢ dense_29/StatefulPartitionedCall¢ dense_30/StatefulPartitionedCall¢"dropout_48/StatefulPartitionedCall¢"dropout_49/StatefulPartitionedCall¢"dropout_50/StatefulPartitionedCall¢#e2e_conv_12/StatefulPartitionedCall¢4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp¢#e2e_conv_13/StatefulPartitionedCall¢4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOpî
#e2e_conv_12/StatefulPartitionedCallStatefulPartitionedCallinputse2e_conv_12_115441*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_e2e_conv_12_layer_call_and_return_conditional_losses_115027
#e2e_conv_13/StatefulPartitionedCallStatefulPartitionedCall,e2e_conv_12/StatefulPartitionedCall:output:0e2e_conv_13_115444*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_e2e_conv_13_layer_call_and_return_conditional_losses_115065®
$Edge-to-Node/StatefulPartitionedCallStatefulPartitionedCall,e2e_conv_13/StatefulPartitionedCall:output:0edge_to_node_115447edge_to_node_115449*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Edge-to-Node_layer_call_and_return_conditional_losses_115086³
%Node-to-Graph/StatefulPartitionedCallStatefulPartitionedCall-Edge-to-Node/StatefulPartitionedCall:output:0node_to_graph_115452node_to_graph_115454*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Node-to-Graph_layer_call_and_return_conditional_losses_115109ü
"dropout_48/StatefulPartitionedCallStatefulPartitionedCall.Node-to-Graph/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_48_layer_call_and_return_conditional_losses_115365á
flatten_16/PartitionedCallPartitionedCall+dropout_48/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_115128ê
concatenate_2/PartitionedCallPartitionedCall#flatten_16/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_115137
 dense_28/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_28_115460dense_28_115462*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_28_layer_call_and_return_conditional_losses_115150
"dropout_49/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0#^dropout_48/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_49_layer_call_and_return_conditional_losses_115319
 dense_29/StatefulPartitionedCallStatefulPartitionedCall+dropout_49/StatefulPartitionedCall:output:0dense_29_115466dense_29_115468*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_29_layer_call_and_return_conditional_losses_115174
"dropout_50/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0#^dropout_49/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_50_layer_call_and_return_conditional_losses_115286
 dense_30/StatefulPartitionedCallStatefulPartitionedCall+dropout_50/StatefulPartitionedCall:output:0dense_30_115472dense_30_115474*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_115198
4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpe2e_conv_12_115441*&
_output_shapes
: *
dtype0
%e2e_conv_12/kernel/Regularizer/SquareSquare<e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$e2e_conv_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¤
"e2e_conv_12/kernel/Regularizer/SumSum)e2e_conv_12/kernel/Regularizer/Square:y:0-e2e_conv_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$e2e_conv_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¦
"e2e_conv_12/kernel/Regularizer/mulMul-e2e_conv_12/kernel/Regularizer/mul/x:output:0+e2e_conv_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpe2e_conv_13_115444*&
_output_shapes
:  *
dtype0
%e2e_conv_13/kernel/Regularizer/SquareSquare<e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  }
$e2e_conv_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¤
"e2e_conv_13/kernel/Regularizer/SumSum)e2e_conv_13/kernel/Regularizer/Square:y:0-e2e_conv_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$e2e_conv_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¦
"e2e_conv_13/kernel/Regularizer/mulMul-e2e_conv_13/kernel/Regularizer/mul/x:output:0+e2e_conv_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOpReadVariableOpedge_to_node_115447*&
_output_shapes
: @*
dtype0 
&Edge-to-Node/kernel/Regularizer/SquareSquare=Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @~
%Edge-to-Node/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             §
#Edge-to-Node/kernel/Regularizer/SumSum*Edge-to-Node/kernel/Regularizer/Square:y:0.Edge-to-Node/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%Edge-to-Node/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#Edge-to-Node/kernel/Regularizer/mulMul.Edge-to-Node/kernel/Regularizer/mul/x:output:0,Edge-to-Node/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOpReadVariableOpnode_to_graph_115452*&
_output_shapes
:@@*
dtype0¢
'Node-to-Graph/kernel/Regularizer/SquareSquare>Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@
&Node-to-Graph/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ª
$Node-to-Graph/kernel/Regularizer/SumSum+Node-to-Graph/kernel/Regularizer/Square:y:0/Node-to-Graph/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: k
&Node-to-Graph/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¬
$Node-to-Graph/kernel/Regularizer/mulMul/Node-to-Graph/kernel/Regularizer/mul/x:output:0-Node-to-Graph/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_30/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp%^Edge-to-Node/StatefulPartitionedCall6^Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp&^Node-to-Graph/StatefulPartitionedCall7^Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall#^dropout_48/StatefulPartitionedCall#^dropout_49/StatefulPartitionedCall#^dropout_50/StatefulPartitionedCall$^e2e_conv_12/StatefulPartitionedCall5^e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp$^e2e_conv_13/StatefulPartitionedCall5^e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2L
$Edge-to-Node/StatefulPartitionedCall$Edge-to-Node/StatefulPartitionedCall2n
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp2N
%Node-to-Graph/StatefulPartitionedCall%Node-to-Graph/StatefulPartitionedCall2p
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2H
"dropout_48/StatefulPartitionedCall"dropout_48/StatefulPartitionedCall2H
"dropout_49/StatefulPartitionedCall"dropout_49/StatefulPartitionedCall2H
"dropout_50/StatefulPartitionedCall"dropout_50/StatefulPartitionedCall2J
#e2e_conv_12/StatefulPartitionedCall#e2e_conv_12/StatefulPartitionedCall2l
4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp4e2e_conv_12/kernel/Regularizer/Square/ReadVariableOp2J
#e2e_conv_13/StatefulPartitionedCall#e2e_conv_13/StatefulPartitionedCall2l
4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

)__inference_dense_28_layer_call_fn_116288

inputs
unknown:	C
	unknown_0:	
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_28_layer_call_and_return_conditional_losses_115150p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿC: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
 
_user_specified_nameinputs
÷
d
+__inference_dropout_50_layer_call_fn_116356

inputs
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_50_layer_call_and_return_conditional_losses_115286p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¿
__inference_loss_fn_1_116415W
=e2e_conv_13_kernel_regularizer_square_readvariableop_resource:  
identity¢4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOpº
4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp=e2e_conv_13_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype0
%e2e_conv_13/kernel/Regularizer/SquareSquare<e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  }
$e2e_conv_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¤
"e2e_conv_13/kernel/Regularizer/SumSum)e2e_conv_13/kernel/Regularizer/Square:y:0-e2e_conv_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$e2e_conv_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:¦
"e2e_conv_13/kernel/Regularizer/mulMul-e2e_conv_13/kernel/Regularizer/mul/x:output:0+e2e_conv_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: d
IdentityIdentity&e2e_conv_13/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: }
NoOpNoOp5^e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2l
4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp4e2e_conv_13/kernel/Regularizer/Square/ReadVariableOp"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ü
serving_defaultè
G
	input_img:
serving_default_input_img:0ÿÿÿÿÿÿÿÿÿ
C
input_struc4
serving_default_input_struc:0ÿÿÿÿÿÿÿÿÿ<
dense_300
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:åâ
Ó
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
±

kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
±

kernel
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
»

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
»

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:_random_generator
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
¥
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U_random_generator
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Xkernel
Ybias
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d_random_generator
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
»

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
oiter

pbeta_1

qbeta_2
	rdecay
slearning_ratemÊmË&mÌ'mÍ.mÎ/mÏImÐJmÑXmÒYmÓgmÔhmÕvÖv×&vØ'vÙ.vÚ/vÛIvÜJvÝXvÞYvßgvàhvá"
	optimizer
v
0
1
&2
'3
.4
/5
I6
J7
X8
Y9
g10
h11"
trackable_list_wrapper
v
0
1
&2
'3
.4
/5
I6
J7
X8
Y9
g10
h11"
trackable_list_wrapper
<
t0
u1
v2
w3"
trackable_list_wrapper
Ê
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ò2ï
)__inference_model_16_layer_call_fn_115256
)__inference_model_16_layer_call_fn_115749
)__inference_model_16_layer_call_fn_115779
)__inference_model_16_layer_call_fn_115559À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_model_16_layer_call_and_return_conditional_losses_115898
D__inference_model_16_layer_call_and_return_conditional_losses_116038
D__inference_model_16_layer_call_and_return_conditional_losses_115624
D__inference_model_16_layer_call_and_return_conditional_losses_115689À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÛBØ
!__inference__wrapped_model_114984	input_imginput_struc"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
}serving_default"
signature_map
,:* 2e2e_conv_12/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
t0"
trackable_list_wrapper
°
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_e2e_conv_12_layer_call_fn_116083¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_e2e_conv_12_layer_call_and_return_conditional_losses_116117¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,:*  2e2e_conv_13/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
u0"
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_e2e_conv_13_layer_call_fn_116130¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_e2e_conv_13_layer_call_and_return_conditional_losses_116164¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
-:+ @2Edge-to-Node/kernel
:@2Edge-to-Node/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
'
v0"
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_Edge-to-Node_layer_call_fn_116179¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_Edge-to-Node_layer_call_and_return_conditional_losses_116196¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.:,@@2Node-to-Graph/kernel
 :@2Node-to-Graph/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
'
w0"
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_Node-to-Graph_layer_call_fn_116211¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_Node-to-Graph_layer_call_and_return_conditional_losses_116228¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
6	variables
7trainable_variables
8regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_dropout_48_layer_call_fn_116233
+__inference_dropout_48_layer_call_fn_116238´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_48_layer_call_and_return_conditional_losses_116243
F__inference_dropout_48_layer_call_and_return_conditional_losses_116255´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_flatten_16_layer_call_fn_116260¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_flatten_16_layer_call_and_return_conditional_losses_116266¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_concatenate_2_layer_call_fn_116272¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_concatenate_2_layer_call_and_return_conditional_losses_116279¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": 	C2dense_28/kernel
:2dense_28/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_28_layer_call_fn_116288¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_28_layer_call_and_return_conditional_losses_116299¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_dropout_49_layer_call_fn_116304
+__inference_dropout_49_layer_call_fn_116309´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_49_layer_call_and_return_conditional_losses_116314
F__inference_dropout_49_layer_call_and_return_conditional_losses_116326´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
#:!
2dense_29/kernel
:2dense_29/bias
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_29_layer_call_fn_116335¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_29_layer_call_and_return_conditional_losses_116346¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
`	variables
atrainable_variables
bregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_dropout_50_layer_call_fn_116351
+__inference_dropout_50_layer_call_fn_116356´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_50_layer_call_and_return_conditional_losses_116361
F__inference_dropout_50_layer_call_and_return_conditional_losses_116373´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
": 	2dense_30/kernel
:2dense_30/bias
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
µnon_trainable_variables
¶layers
·metrics
 ¸layer_regularization_losses
¹layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_30_layer_call_fn_116382¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_30_layer_call_and_return_conditional_losses_116393¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
³2°
__inference_loss_fn_0_116404
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_1_116415
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_2_116426
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_3_116437
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
 "
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
8
º0
»1
¼2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ØBÕ
$__inference_signature_wrapper_116070	input_imginput_struc"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
t0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
u0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
v0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
w0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

½total

¾count
¿	variables
À	keras_api"
_tf_keras_metric
c

Átotal

Âcount
Ã
_fn_kwargs
Ä	variables
Å	keras_api"
_tf_keras_metric
]
Æ
thresholds
Çaccumulator
È	variables
É	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
½0
¾1"
trackable_list_wrapper
.
¿	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Á0
Â1"
trackable_list_wrapper
.
Ä	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
Ç0"
trackable_list_wrapper
.
È	variables"
_generic_user_object
1:/ 2Adam/e2e_conv_12/kernel/m
1:/  2Adam/e2e_conv_13/kernel/m
2:0 @2Adam/Edge-to-Node/kernel/m
$:"@2Adam/Edge-to-Node/bias/m
3:1@@2Adam/Node-to-Graph/kernel/m
%:#@2Adam/Node-to-Graph/bias/m
':%	C2Adam/dense_28/kernel/m
!:2Adam/dense_28/bias/m
(:&
2Adam/dense_29/kernel/m
!:2Adam/dense_29/bias/m
':%	2Adam/dense_30/kernel/m
 :2Adam/dense_30/bias/m
1:/ 2Adam/e2e_conv_12/kernel/v
1:/  2Adam/e2e_conv_13/kernel/v
2:0 @2Adam/Edge-to-Node/kernel/v
$:"@2Adam/Edge-to-Node/bias/v
3:1@@2Adam/Node-to-Graph/kernel/v
%:#@2Adam/Node-to-Graph/bias/v
':%	C2Adam/dense_28/kernel/v
!:2Adam/dense_28/bias/v
(:&
2Adam/dense_29/kernel/v
!:2Adam/dense_29/bias/v
':%	2Adam/dense_30/kernel/v
 :2Adam/dense_30/bias/v¸
H__inference_Edge-to-Node_layer_call_and_return_conditional_losses_116196l&'7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
-__inference_Edge-to-Node_layer_call_fn_116179_&'7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ@¹
I__inference_Node-to-Graph_layer_call_and_return_conditional_losses_116228l./7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
.__inference_Node-to-Graph_layer_call_fn_116211_./7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@Ñ
!__inference__wrapped_model_114984«&'./IJXYghf¢c
\¢Y
WT
+(
	input_imgÿÿÿÿÿÿÿÿÿ
%"
input_strucÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_30"
dense_30ÿÿÿÿÿÿÿÿÿÑ
I__inference_concatenate_2_layer_call_and_return_conditional_losses_116279Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ@
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿC
 ¨
.__inference_concatenate_2_layer_call_fn_116272vZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ@
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿC¥
D__inference_dense_28_layer_call_and_return_conditional_losses_116299]IJ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿC
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
)__inference_dense_28_layer_call_fn_116288PIJ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿC
ª "ÿÿÿÿÿÿÿÿÿ¦
D__inference_dense_29_layer_call_and_return_conditional_losses_116346^XY0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dense_29_layer_call_fn_116335QXY0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
D__inference_dense_30_layer_call_and_return_conditional_losses_116393]gh0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
)__inference_dense_30_layer_call_fn_116382Pgh0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¶
F__inference_dropout_48_layer_call_and_return_conditional_losses_116243l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 ¶
F__inference_dropout_48_layer_call_and_return_conditional_losses_116255l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
+__inference_dropout_48_layer_call_fn_116233_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@
+__inference_dropout_48_layer_call_fn_116238_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@¨
F__inference_dropout_49_layer_call_and_return_conditional_losses_116314^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¨
F__inference_dropout_49_layer_call_and_return_conditional_losses_116326^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dropout_49_layer_call_fn_116304Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_dropout_49_layer_call_fn_116309Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¨
F__inference_dropout_50_layer_call_and_return_conditional_losses_116361^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¨
F__inference_dropout_50_layer_call_and_return_conditional_losses_116373^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dropout_50_layer_call_fn_116351Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_dropout_50_layer_call_fn_116356Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¶
G__inference_e2e_conv_12_layer_call_and_return_conditional_losses_116117k7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_e2e_conv_12_layer_call_fn_116083^7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ ¶
G__inference_e2e_conv_13_layer_call_and_return_conditional_losses_116164k7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_e2e_conv_13_layer_call_fn_116130^7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ ª
F__inference_flatten_16_layer_call_and_return_conditional_losses_116266`7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
+__inference_flatten_16_layer_call_fn_116260S7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@;
__inference_loss_fn_0_116404¢

¢ 
ª " ;
__inference_loss_fn_1_116415¢

¢ 
ª " ;
__inference_loss_fn_2_116426&¢

¢ 
ª " ;
__inference_loss_fn_3_116437.¢

¢ 
ª " î
D__inference_model_16_layer_call_and_return_conditional_losses_115624¥&'./IJXYghn¢k
d¢a
WT
+(
	input_imgÿÿÿÿÿÿÿÿÿ
%"
input_strucÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 î
D__inference_model_16_layer_call_and_return_conditional_losses_115689¥&'./IJXYghn¢k
d¢a
WT
+(
	input_imgÿÿÿÿÿÿÿÿÿ
%"
input_strucÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ê
D__inference_model_16_layer_call_and_return_conditional_losses_115898¡&'./IJXYghj¢g
`¢]
SP
*'
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ê
D__inference_model_16_layer_call_and_return_conditional_losses_116038¡&'./IJXYghj¢g
`¢]
SP
*'
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Æ
)__inference_model_16_layer_call_fn_115256&'./IJXYghn¢k
d¢a
WT
+(
	input_imgÿÿÿÿÿÿÿÿÿ
%"
input_strucÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÆ
)__inference_model_16_layer_call_fn_115559&'./IJXYghn¢k
d¢a
WT
+(
	input_imgÿÿÿÿÿÿÿÿÿ
%"
input_strucÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÂ
)__inference_model_16_layer_call_fn_115749&'./IJXYghj¢g
`¢]
SP
*'
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÂ
)__inference_model_16_layer_call_fn_115779&'./IJXYghj¢g
`¢]
SP
*'
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿë
$__inference_signature_wrapper_116070Â&'./IJXYgh}¢z
¢ 
sªp
8
	input_img+(
	input_imgÿÿÿÿÿÿÿÿÿ
4
input_struc%"
input_strucÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_30"
dense_30ÿÿÿÿÿÿÿÿÿ