§¿!
õ
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
ú
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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
 "serve*2.8.02v2.8.0-0-g3f878cff5b68âú

e2e_conv_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ö*"
shared_namee2e_conv_4/kernel

%e2e_conv_4/kernel/Read/ReadVariableOpReadVariableOpe2e_conv_4/kernel*'
_output_shapes
:ö*
dtype0

batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_6/gamma

/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
:*
dtype0

batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_6/beta

.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:*
dtype0

!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_6/moving_mean

5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:*
dtype0
¢
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_6/moving_variance

9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:*
dtype0

e2e_conv_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ö*"
shared_namee2e_conv_5/kernel

%e2e_conv_5/kernel/Read/ReadVariableOpReadVariableOpe2e_conv_5/kernel*'
_output_shapes
:ö*
dtype0

batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_7/gamma

/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:*
dtype0

batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_7/beta

.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:*
dtype0

!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_7/moving_mean

5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:*
dtype0
¢
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_7/moving_variance

9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:*
dtype0

Edge-to-Node/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ö*$
shared_nameEdge-to-Node/kernel

'Edge-to-Node/kernel/Read/ReadVariableOpReadVariableOpEdge-to-Node/kernel*'
_output_shapes
:ö*
dtype0
z
Edge-to-Node/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameEdge-to-Node/bias
s
%Edge-to-Node/bias/Read/ReadVariableOpReadVariableOpEdge-to-Node/bias*
_output_shapes
:*
dtype0

batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_8/gamma

/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
:*
dtype0

batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_8/beta

.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
:*
dtype0

!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_8/moving_mean

5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
:*
dtype0
¢
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_8/moving_variance

9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
:*
dtype0

Node-to-Graph/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ö *%
shared_nameNode-to-Graph/kernel

(Node-to-Graph/kernel/Read/ReadVariableOpReadVariableOpNode-to-Graph/kernel*'
_output_shapes
:ö *
dtype0
|
Node-to-Graph/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameNode-to-Graph/bias
u
&Node-to-Graph/bias/Read/ReadVariableOpReadVariableOpNode-to-Graph/bias*
_output_shapes
: *
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#@*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:#@*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:@*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:@@*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:@*
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:@*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
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

Nadam/e2e_conv_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ö**
shared_nameNadam/e2e_conv_4/kernel/m

-Nadam/e2e_conv_4/kernel/m/Read/ReadVariableOpReadVariableOpNadam/e2e_conv_4/kernel/m*'
_output_shapes
:ö*
dtype0

#Nadam/batch_normalization_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Nadam/batch_normalization_6/gamma/m

7Nadam/batch_normalization_6/gamma/m/Read/ReadVariableOpReadVariableOp#Nadam/batch_normalization_6/gamma/m*
_output_shapes
:*
dtype0

"Nadam/batch_normalization_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Nadam/batch_normalization_6/beta/m

6Nadam/batch_normalization_6/beta/m/Read/ReadVariableOpReadVariableOp"Nadam/batch_normalization_6/beta/m*
_output_shapes
:*
dtype0

Nadam/e2e_conv_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ö**
shared_nameNadam/e2e_conv_5/kernel/m

-Nadam/e2e_conv_5/kernel/m/Read/ReadVariableOpReadVariableOpNadam/e2e_conv_5/kernel/m*'
_output_shapes
:ö*
dtype0

#Nadam/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Nadam/batch_normalization_7/gamma/m

7Nadam/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp#Nadam/batch_normalization_7/gamma/m*
_output_shapes
:*
dtype0

"Nadam/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Nadam/batch_normalization_7/beta/m

6Nadam/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp"Nadam/batch_normalization_7/beta/m*
_output_shapes
:*
dtype0

Nadam/Edge-to-Node/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ö*,
shared_nameNadam/Edge-to-Node/kernel/m

/Nadam/Edge-to-Node/kernel/m/Read/ReadVariableOpReadVariableOpNadam/Edge-to-Node/kernel/m*'
_output_shapes
:ö*
dtype0

Nadam/Edge-to-Node/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameNadam/Edge-to-Node/bias/m

-Nadam/Edge-to-Node/bias/m/Read/ReadVariableOpReadVariableOpNadam/Edge-to-Node/bias/m*
_output_shapes
:*
dtype0

#Nadam/batch_normalization_8/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Nadam/batch_normalization_8/gamma/m

7Nadam/batch_normalization_8/gamma/m/Read/ReadVariableOpReadVariableOp#Nadam/batch_normalization_8/gamma/m*
_output_shapes
:*
dtype0

"Nadam/batch_normalization_8/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Nadam/batch_normalization_8/beta/m

6Nadam/batch_normalization_8/beta/m/Read/ReadVariableOpReadVariableOp"Nadam/batch_normalization_8/beta/m*
_output_shapes
:*
dtype0

Nadam/Node-to-Graph/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ö *-
shared_nameNadam/Node-to-Graph/kernel/m

0Nadam/Node-to-Graph/kernel/m/Read/ReadVariableOpReadVariableOpNadam/Node-to-Graph/kernel/m*'
_output_shapes
:ö *
dtype0

Nadam/Node-to-Graph/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameNadam/Node-to-Graph/bias/m

.Nadam/Node-to-Graph/bias/m/Read/ReadVariableOpReadVariableOpNadam/Node-to-Graph/bias/m*
_output_shapes
: *
dtype0

Nadam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#@*'
shared_nameNadam/dense_6/kernel/m

*Nadam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_6/kernel/m*
_output_shapes

:#@*
dtype0

Nadam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameNadam/dense_6/bias/m
y
(Nadam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_6/bias/m*
_output_shapes
:@*
dtype0

Nadam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameNadam/dense_7/kernel/m

*Nadam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_7/kernel/m*
_output_shapes

:@@*
dtype0

Nadam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameNadam/dense_7/bias/m
y
(Nadam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_7/bias/m*
_output_shapes
:@*
dtype0

Nadam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameNadam/dense_8/kernel/m

*Nadam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_8/kernel/m*
_output_shapes

:@*
dtype0

Nadam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/dense_8/bias/m
y
(Nadam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_8/bias/m*
_output_shapes
:*
dtype0

Nadam/e2e_conv_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ö**
shared_nameNadam/e2e_conv_4/kernel/v

-Nadam/e2e_conv_4/kernel/v/Read/ReadVariableOpReadVariableOpNadam/e2e_conv_4/kernel/v*'
_output_shapes
:ö*
dtype0

#Nadam/batch_normalization_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Nadam/batch_normalization_6/gamma/v

7Nadam/batch_normalization_6/gamma/v/Read/ReadVariableOpReadVariableOp#Nadam/batch_normalization_6/gamma/v*
_output_shapes
:*
dtype0

"Nadam/batch_normalization_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Nadam/batch_normalization_6/beta/v

6Nadam/batch_normalization_6/beta/v/Read/ReadVariableOpReadVariableOp"Nadam/batch_normalization_6/beta/v*
_output_shapes
:*
dtype0

Nadam/e2e_conv_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ö**
shared_nameNadam/e2e_conv_5/kernel/v

-Nadam/e2e_conv_5/kernel/v/Read/ReadVariableOpReadVariableOpNadam/e2e_conv_5/kernel/v*'
_output_shapes
:ö*
dtype0

#Nadam/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Nadam/batch_normalization_7/gamma/v

7Nadam/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp#Nadam/batch_normalization_7/gamma/v*
_output_shapes
:*
dtype0

"Nadam/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Nadam/batch_normalization_7/beta/v

6Nadam/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp"Nadam/batch_normalization_7/beta/v*
_output_shapes
:*
dtype0

Nadam/Edge-to-Node/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ö*,
shared_nameNadam/Edge-to-Node/kernel/v

/Nadam/Edge-to-Node/kernel/v/Read/ReadVariableOpReadVariableOpNadam/Edge-to-Node/kernel/v*'
_output_shapes
:ö*
dtype0

Nadam/Edge-to-Node/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameNadam/Edge-to-Node/bias/v

-Nadam/Edge-to-Node/bias/v/Read/ReadVariableOpReadVariableOpNadam/Edge-to-Node/bias/v*
_output_shapes
:*
dtype0

#Nadam/batch_normalization_8/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Nadam/batch_normalization_8/gamma/v

7Nadam/batch_normalization_8/gamma/v/Read/ReadVariableOpReadVariableOp#Nadam/batch_normalization_8/gamma/v*
_output_shapes
:*
dtype0

"Nadam/batch_normalization_8/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Nadam/batch_normalization_8/beta/v

6Nadam/batch_normalization_8/beta/v/Read/ReadVariableOpReadVariableOp"Nadam/batch_normalization_8/beta/v*
_output_shapes
:*
dtype0

Nadam/Node-to-Graph/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ö *-
shared_nameNadam/Node-to-Graph/kernel/v

0Nadam/Node-to-Graph/kernel/v/Read/ReadVariableOpReadVariableOpNadam/Node-to-Graph/kernel/v*'
_output_shapes
:ö *
dtype0

Nadam/Node-to-Graph/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameNadam/Node-to-Graph/bias/v

.Nadam/Node-to-Graph/bias/v/Read/ReadVariableOpReadVariableOpNadam/Node-to-Graph/bias/v*
_output_shapes
: *
dtype0

Nadam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#@*'
shared_nameNadam/dense_6/kernel/v

*Nadam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_6/kernel/v*
_output_shapes

:#@*
dtype0

Nadam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameNadam/dense_6/bias/v
y
(Nadam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_6/bias/v*
_output_shapes
:@*
dtype0

Nadam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameNadam/dense_7/kernel/v

*Nadam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_7/kernel/v*
_output_shapes

:@@*
dtype0

Nadam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameNadam/dense_7/bias/v
y
(Nadam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_7/bias/v*
_output_shapes
:@*
dtype0

Nadam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameNadam/dense_8/kernel/v

*Nadam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_8/kernel/v*
_output_shapes

:@*
dtype0

Nadam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/dense_8/bias/v
y
(Nadam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_8/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ð
valueÅBÁ B¹
´
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer-15
layer_with_weights-9
layer-16
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 


kernel
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*
Õ
"axis
	#gamma
$beta
%moving_mean
&moving_variance
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses*


-kernel
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses*
Õ
4axis
	5gamma
6beta
7moving_mean
8moving_variance
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses*
¦

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses*
Õ
Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses*
¦

Rkernel
Sbias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses*
¥
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^_random_generator
___call__
*`&call_and_return_all_conditional_losses* 

a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses* 
* 

g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses* 
¦

mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses*
¥
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y_random_generator
z__call__
*{&call_and_return_all_conditional_losses* 
ª

|kernel
}bias
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¬
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
Ê
	iter
beta_1
beta_2

decay
learning_rate
momentum_cachemþ#mÿ$m-m5m6m?m@mHmImRmSmmmnm|m}m	m	mv#v$v-v5v6v?v@vHvIvRvSvmvnv|v}v	v 	v¡*
¼
0
#1
$2
%3
&4
-5
56
67
78
89
?10
@11
H12
I13
J14
K15
R16
S17
m18
n19
|20
}21
22
23*

0
#1
$2
-3
54
65
?6
@7
H8
I9
R10
S11
m12
n13
|14
}15
16
17*
"
0
1
2
3* 
µ
non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

¢serving_default* 
a[
VARIABLE_VALUEe2e_conv_4/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*


0* 

£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_6/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_6/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_6/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_6/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
#0
$1
%2
&3*

#0
$1*
* 

¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*
* 
* 
a[
VARIABLE_VALUEe2e_conv_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*

-0*

-0*


0* 

­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_7/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_7/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_7/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_7/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
50
61
72
83*

50
61*
* 

²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUEEdge-to-Node/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEEdge-to-Node/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
@1*

?0
@1*


0* 

·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_8/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_8/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_8/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_8/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
H0
I1
J2
K3*

H0
I1*
* 

¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*
* 
* 
d^
VARIABLE_VALUENode-to-Graph/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUENode-to-Graph/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

R0
S1*

R0
S1*


0* 

Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses* 
* 
* 
^X
VARIABLE_VALUEdense_6/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_6/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

m0
n1*

m0
n1*
* 

Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 
* 
* 
* 
^X
VARIABLE_VALUEdense_7/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_7/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

|0
}1*

|0
}1*
* 

ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
^X
VARIABLE_VALUEdense_8/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_8/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

énon_trainable_variables
êlayers
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
MG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
.
%0
&1
72
83
J4
K5*

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
13
14
15
16*

î0
ï1
ð2*
* 
* 
* 
* 
* 
* 


0* 
* 

%0
&1*
* 
* 
* 
* 
* 
* 
* 


0* 
* 

70
81*
* 
* 
* 
* 
* 
* 
* 


0* 
* 

J0
K1*
* 
* 
* 
* 
* 
* 
* 


0* 
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

ñtotal

òcount
ó	variables
ô	keras_api*
M

õtotal

öcount
÷
_fn_kwargs
ø	variables
ù	keras_api*
G
ú
thresholds
ûaccumulator
ü	variables
ý	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

ñ0
ò1*

ó	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

õ0
ö1*

ø	variables*
* 
_Y
VARIABLE_VALUEaccumulator:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUE*

û0*

ü	variables*

VARIABLE_VALUENadam/e2e_conv_4/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Nadam/batch_normalization_6/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Nadam/batch_normalization_6/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/e2e_conv_5/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Nadam/batch_normalization_7/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Nadam/batch_normalization_7/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Edge-to-Node/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUENadam/Edge-to-Node/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Nadam/batch_normalization_8/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Nadam/batch_normalization_8/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Node-to-Graph/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUENadam/Node-to-Graph/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUENadam/dense_6/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_6/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUENadam/dense_7/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_7/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUENadam/dense_8/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_8/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/e2e_conv_4/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Nadam/batch_normalization_6/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Nadam/batch_normalization_6/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/e2e_conv_5/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Nadam/batch_normalization_7/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Nadam/batch_normalization_7/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Edge-to-Node/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUENadam/Edge-to-Node/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Nadam/batch_normalization_8/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Nadam/batch_normalization_8/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Node-to-Graph/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUENadam/Node-to-Graph/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUENadam/dense_6/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_6/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUENadam/dense_7/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_7/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUENadam/dense_8/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_8/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_imgPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿöö
~
serving_default_input_strucPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ß
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_imgserving_default_input_struce2e_conv_4/kernelbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancee2e_conv_5/kernelbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceEdge-to-Node/kernelEdge-to-Node/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceNode-to-Graph/kernelNode-to-Graph/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/bias*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_33625
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ò
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%e2e_conv_4/kernel/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp%e2e_conv_5/kernel/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp'Edge-to-Node/kernel/Read/ReadVariableOp%Edge-to-Node/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp(Node-to-Graph/kernel/Read/ReadVariableOp&Node-to-Graph/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpaccumulator/Read/ReadVariableOp-Nadam/e2e_conv_4/kernel/m/Read/ReadVariableOp7Nadam/batch_normalization_6/gamma/m/Read/ReadVariableOp6Nadam/batch_normalization_6/beta/m/Read/ReadVariableOp-Nadam/e2e_conv_5/kernel/m/Read/ReadVariableOp7Nadam/batch_normalization_7/gamma/m/Read/ReadVariableOp6Nadam/batch_normalization_7/beta/m/Read/ReadVariableOp/Nadam/Edge-to-Node/kernel/m/Read/ReadVariableOp-Nadam/Edge-to-Node/bias/m/Read/ReadVariableOp7Nadam/batch_normalization_8/gamma/m/Read/ReadVariableOp6Nadam/batch_normalization_8/beta/m/Read/ReadVariableOp0Nadam/Node-to-Graph/kernel/m/Read/ReadVariableOp.Nadam/Node-to-Graph/bias/m/Read/ReadVariableOp*Nadam/dense_6/kernel/m/Read/ReadVariableOp(Nadam/dense_6/bias/m/Read/ReadVariableOp*Nadam/dense_7/kernel/m/Read/ReadVariableOp(Nadam/dense_7/bias/m/Read/ReadVariableOp*Nadam/dense_8/kernel/m/Read/ReadVariableOp(Nadam/dense_8/bias/m/Read/ReadVariableOp-Nadam/e2e_conv_4/kernel/v/Read/ReadVariableOp7Nadam/batch_normalization_6/gamma/v/Read/ReadVariableOp6Nadam/batch_normalization_6/beta/v/Read/ReadVariableOp-Nadam/e2e_conv_5/kernel/v/Read/ReadVariableOp7Nadam/batch_normalization_7/gamma/v/Read/ReadVariableOp6Nadam/batch_normalization_7/beta/v/Read/ReadVariableOp/Nadam/Edge-to-Node/kernel/v/Read/ReadVariableOp-Nadam/Edge-to-Node/bias/v/Read/ReadVariableOp7Nadam/batch_normalization_8/gamma/v/Read/ReadVariableOp6Nadam/batch_normalization_8/beta/v/Read/ReadVariableOp0Nadam/Node-to-Graph/kernel/v/Read/ReadVariableOp.Nadam/Node-to-Graph/bias/v/Read/ReadVariableOp*Nadam/dense_6/kernel/v/Read/ReadVariableOp(Nadam/dense_6/bias/v/Read/ReadVariableOp*Nadam/dense_7/kernel/v/Read/ReadVariableOp(Nadam/dense_7/bias/v/Read/ReadVariableOp*Nadam/dense_8/kernel/v/Read/ReadVariableOp(Nadam/dense_8/bias/v/Read/ReadVariableOpConst*T
TinM
K2I	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_34415
á
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamee2e_conv_4/kernelbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancee2e_conv_5/kernelbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceEdge-to-Node/kernelEdge-to-Node/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceNode-to-Graph/kernelNode-to-Graph/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcounttotal_1count_1accumulatorNadam/e2e_conv_4/kernel/m#Nadam/batch_normalization_6/gamma/m"Nadam/batch_normalization_6/beta/mNadam/e2e_conv_5/kernel/m#Nadam/batch_normalization_7/gamma/m"Nadam/batch_normalization_7/beta/mNadam/Edge-to-Node/kernel/mNadam/Edge-to-Node/bias/m#Nadam/batch_normalization_8/gamma/m"Nadam/batch_normalization_8/beta/mNadam/Node-to-Graph/kernel/mNadam/Node-to-Graph/bias/mNadam/dense_6/kernel/mNadam/dense_6/bias/mNadam/dense_7/kernel/mNadam/dense_7/bias/mNadam/dense_8/kernel/mNadam/dense_8/bias/mNadam/e2e_conv_4/kernel/v#Nadam/batch_normalization_6/gamma/v"Nadam/batch_normalization_6/beta/vNadam/e2e_conv_5/kernel/v#Nadam/batch_normalization_7/gamma/v"Nadam/batch_normalization_7/beta/vNadam/Edge-to-Node/kernel/vNadam/Edge-to-Node/bias/v#Nadam/batch_normalization_8/gamma/v"Nadam/batch_normalization_8/beta/vNadam/Node-to-Graph/kernel/vNadam/Node-to-Graph/bias/vNadam/dense_6/kernel/vNadam/dense_6/bias/vNadam/dense_7/kernel/vNadam/dense_7/bias/vNadam/dense_8/kernel/vNadam/dense_8/bias/v*S
TinL
J2H*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_34638³
	
Ð
5__inference_batch_normalization_7_layer_call_fn_33794

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_32073
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

F__inference_BrainNetCNN_layer_call_and_return_conditional_losses_33387
inputs_0
inputs_1=
"e2e_conv_4_readvariableop_resource:ö;
-batch_normalization_6_readvariableop_resource:=
/batch_normalization_6_readvariableop_1_resource:L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:=
"e2e_conv_5_readvariableop_resource:ö;
-batch_normalization_7_readvariableop_resource:=
/batch_normalization_7_readvariableop_1_resource:L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:F
+edge_to_node_conv2d_readvariableop_resource:ö:
,edge_to_node_biasadd_readvariableop_resource:;
-batch_normalization_8_readvariableop_resource:=
/batch_normalization_8_readvariableop_1_resource:L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:G
,node_to_graph_conv2d_readvariableop_resource:ö ;
-node_to_graph_biasadd_readvariableop_resource: 8
&dense_6_matmul_readvariableop_resource:#@5
'dense_6_biasadd_readvariableop_resource:@8
&dense_7_matmul_readvariableop_resource:@@5
'dense_7_biasadd_readvariableop_resource:@8
&dense_8_matmul_readvariableop_resource:@5
'dense_8_biasadd_readvariableop_resource:
identity¢#Edge-to-Node/BiasAdd/ReadVariableOp¢"Edge-to-Node/Conv2D/ReadVariableOp¢5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp¢$Node-to-Graph/BiasAdd/ReadVariableOp¢#Node-to-Graph/Conv2D/ReadVariableOp¢6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp¢5batch_normalization_6/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_6/ReadVariableOp¢&batch_normalization_6/ReadVariableOp_1¢5batch_normalization_7/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_7/ReadVariableOp¢&batch_normalization_7/ReadVariableOp_1¢5batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_8/ReadVariableOp¢&batch_normalization_8/ReadVariableOp_1¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢e2e_conv_4/ReadVariableOp¢e2e_conv_4/ReadVariableOp_1¢3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp¢e2e_conv_5/ReadVariableOp¢e2e_conv_5/ReadVariableOp_1¢3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp
e2e_conv_4/ReadVariableOpReadVariableOp"e2e_conv_4_readvariableop_resource*'
_output_shapes
:ö*
dtype0o
e2e_conv_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        q
 e2e_conv_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       q
 e2e_conv_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¿
e2e_conv_4/strided_sliceStridedSlice!e2e_conv_4/ReadVariableOp:value:0'e2e_conv_4/strided_slice/stack:output:0)e2e_conv_4/strided_slice/stack_1:output:0)e2e_conv_4/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_maskq
e2e_conv_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   ö         
e2e_conv_4/ReshapeReshape!e2e_conv_4/strided_slice:output:0!e2e_conv_4/Reshape/shape:output:0*
T0*'
_output_shapes
:ö
e2e_conv_4/ReadVariableOp_1ReadVariableOp"e2e_conv_4_readvariableop_resource*'
_output_shapes
:ö*
dtype0q
 e2e_conv_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"e2e_conv_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"e2e_conv_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      É
e2e_conv_4/strided_slice_1StridedSlice#e2e_conv_4/ReadVariableOp_1:value:0)e2e_conv_4/strided_slice_1/stack:output:0+e2e_conv_4/strided_slice_1/stack_1:output:0+e2e_conv_4/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_masks
e2e_conv_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ö            
e2e_conv_4/Reshape_1Reshape#e2e_conv_4/strided_slice_1:output:0#e2e_conv_4/Reshape_1/shape:output:0*
T0*'
_output_shapes
:ö«
e2e_conv_4/convolutionConv2Dinputs_0e2e_conv_4/Reshape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides
¯
e2e_conv_4/convolution_1Conv2Dinputs_0e2e_conv_4/Reshape_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides
X
e2e_conv_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¦D
e2e_conv_4/concatConcatV2!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0e2e_conv_4/concat/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿööZ
e2e_conv_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :¾@
e2e_conv_4/concat_1ConcatV2e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0!e2e_conv_4/concat_1/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
e2e_conv_4/addAddV2e2e_conv_4/concat:output:0e2e_conv_4/concat_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype0
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype0°
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0´
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0²
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3e2e_conv_4/add:z:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿöö:::::*
epsilon%o:*
is_training( 
e2e_conv_5/ReadVariableOpReadVariableOp"e2e_conv_5_readvariableop_resource*'
_output_shapes
:ö*
dtype0o
e2e_conv_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        q
 e2e_conv_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       q
 e2e_conv_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¿
e2e_conv_5/strided_sliceStridedSlice!e2e_conv_5/ReadVariableOp:value:0'e2e_conv_5/strided_slice/stack:output:0)e2e_conv_5/strided_slice/stack_1:output:0)e2e_conv_5/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_maskq
e2e_conv_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   ö         
e2e_conv_5/ReshapeReshape!e2e_conv_5/strided_slice:output:0!e2e_conv_5/Reshape/shape:output:0*
T0*'
_output_shapes
:ö
e2e_conv_5/ReadVariableOp_1ReadVariableOp"e2e_conv_5_readvariableop_resource*'
_output_shapes
:ö*
dtype0q
 e2e_conv_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"e2e_conv_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"e2e_conv_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      É
e2e_conv_5/strided_slice_1StridedSlice#e2e_conv_5/ReadVariableOp_1:value:0)e2e_conv_5/strided_slice_1/stack:output:0+e2e_conv_5/strided_slice_1/stack_1:output:0+e2e_conv_5/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_masks
e2e_conv_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ö            
e2e_conv_5/Reshape_1Reshape#e2e_conv_5/strided_slice_1:output:0#e2e_conv_5/Reshape_1/shape:output:0*
T0*'
_output_shapes
:öÍ
e2e_conv_5/convolutionConv2D*batch_normalization_6/FusedBatchNormV3:y:0e2e_conv_5/Reshape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides
Ñ
e2e_conv_5/convolution_1Conv2D*batch_normalization_6/FusedBatchNormV3:y:0e2e_conv_5/Reshape_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides
X
e2e_conv_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¦D
e2e_conv_5/concatConcatV2!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0e2e_conv_5/concat/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿööZ
e2e_conv_5/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :¾@
e2e_conv_5/concat_1ConcatV2e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0!e2e_conv_5/concat_1/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
e2e_conv_5/addAddV2e2e_conv_5/concat:output:0e2e_conv_5/concat_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype0
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype0°
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0´
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0²
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3e2e_conv_5/add:z:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿöö:::::*
epsilon%o:*
is_training( 
"Edge-to-Node/Conv2D/ReadVariableOpReadVariableOp+edge_to_node_conv2d_readvariableop_resource*'
_output_shapes
:ö*
dtype0Ù
Edge-to-Node/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0*Edge-to-Node/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides

#Edge-to-Node/BiasAdd/ReadVariableOpReadVariableOp,edge_to_node_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¥
Edge-to-Node/BiasAddBiasAddEdge-to-Node/Conv2D:output:0+Edge-to-Node/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿös
Edge-to-Node/ReluReluEdge-to-Node/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0°
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0´
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0¾
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3Edge-to-Node/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿö:::::*
epsilon%o:*
is_training( 
#Node-to-Graph/Conv2D/ReadVariableOpReadVariableOp,node_to_graph_conv2d_readvariableop_resource*'
_output_shapes
:ö *
dtype0Ú
Node-to-Graph/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0+Node-to-Graph/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

$Node-to-Graph/BiasAdd/ReadVariableOpReadVariableOp-node_to_graph_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0§
Node-to-Graph/BiasAddBiasAddNode-to-Graph/Conv2D:output:0,Node-to-Graph/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
Node-to-Graph/ReluReluNode-to-Graph/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
dropout_6/IdentityIdentity Node-to-Graph/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
flatten_2/ReshapeReshapedropout_6/Identity:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¥
concatenate_2/concatConcatV2flatten_2/Reshape:output:0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:#@*
dtype0
dense_6/MatMulMatMulconcatenate_2/concat:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
dropout_7/IdentityIdentitydense_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0
dense_7/MatMulMatMuldropout_7/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
dropout_8/IdentityIdentitydense_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_8/MatMulMatMuldropout_8/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_8/SigmoidSigmoiddense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"e2e_conv_4_readvariableop_resource*'
_output_shapes
:ö*
dtype0
$e2e_conv_4/kernel/Regularizer/SquareSquare;e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö|
#e2e_conv_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¡
!e2e_conv_4/kernel/Regularizer/SumSum(e2e_conv_4/kernel/Regularizer/Square:y:0,e2e_conv_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#e2e_conv_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<£
!e2e_conv_4/kernel/Regularizer/mulMul,e2e_conv_4/kernel/Regularizer/mul/x:output:0*e2e_conv_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"e2e_conv_5_readvariableop_resource*'
_output_shapes
:ö*
dtype0
$e2e_conv_5/kernel/Regularizer/SquareSquare;e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö|
#e2e_conv_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¡
!e2e_conv_5/kernel/Regularizer/SumSum(e2e_conv_5/kernel/Regularizer/Square:y:0,e2e_conv_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#e2e_conv_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<£
!e2e_conv_5/kernel/Regularizer/mulMul,e2e_conv_5/kernel/Regularizer/mul/x:output:0*e2e_conv_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ª
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+edge_to_node_conv2d_readvariableop_resource*'
_output_shapes
:ö*
dtype0¡
&Edge-to-Node/kernel/Regularizer/SquareSquare=Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö~
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
 *
×#<©
#Edge-to-Node/kernel/Regularizer/mulMul.Edge-to-Node/kernel/Regularizer/mul/x:output:0,Edge-to-Node/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¬
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOpReadVariableOp,node_to_graph_conv2d_readvariableop_resource*'
_output_shapes
:ö *
dtype0£
'Node-to-Graph/kernel/Regularizer/SquareSquare>Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö 
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
 *
×#<¬
$Node-to-Graph/kernel/Regularizer/mulMul/Node-to-Graph/kernel/Regularizer/mul/x:output:0-Node-to-Graph/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentitydense_8/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸

NoOpNoOp$^Edge-to-Node/BiasAdd/ReadVariableOp#^Edge-to-Node/Conv2D/ReadVariableOp6^Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp%^Node-to-Graph/BiasAdd/ReadVariableOp$^Node-to-Graph/Conv2D/ReadVariableOp7^Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp6^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^e2e_conv_4/ReadVariableOp^e2e_conv_4/ReadVariableOp_14^e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp^e2e_conv_5/ReadVariableOp^e2e_conv_5/ReadVariableOp_14^e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿöö:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 2J
#Edge-to-Node/BiasAdd/ReadVariableOp#Edge-to-Node/BiasAdd/ReadVariableOp2H
"Edge-to-Node/Conv2D/ReadVariableOp"Edge-to-Node/Conv2D/ReadVariableOp2n
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp2L
$Node-to-Graph/BiasAdd/ReadVariableOp$Node-to-Graph/BiasAdd/ReadVariableOp2J
#Node-to-Graph/Conv2D/ReadVariableOp#Node-to-Graph/Conv2D/ReadVariableOp2p
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp2n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp26
e2e_conv_4/ReadVariableOpe2e_conv_4/ReadVariableOp2:
e2e_conv_4/ReadVariableOp_1e2e_conv_4/ReadVariableOp_12j
3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp26
e2e_conv_5/ReadVariableOpe2e_conv_5/ReadVariableOp2:
e2e_conv_5/ReadVariableOp_1e2e_conv_5/ReadVariableOp_12j
3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
	
Ð
5__inference_batch_normalization_7_layer_call_fn_33807

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_32104
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
¢
,__inference_Edge-to-Node_layer_call_fn_33858

inputs"
unknown:ö
	unknown_0:
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_Edge-to-Node_layer_call_and_return_conditional_losses_32299x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿöö: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
 
_user_specified_nameinputs
²

c
D__inference_dropout_6_layer_call_and_return_conditional_losses_33996

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


#__inference_signature_wrapper_33625
	input_img
input_struc"
unknown:ö
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:$
	unknown_4:ö
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:$
	unknown_9:ö

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:%

unknown_15:ö 

unknown_16: 

unknown_17:#@

unknown_18:@

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCall	input_imginput_strucunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_31987o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿöö:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
#
_user_specified_name	input_img:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_struc
Ä
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_34007

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ù
¿
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_32168

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ó
B__inference_dense_6_layer_call_and_return_conditional_losses_34040

inputs0
matmul_readvariableop_resource:#@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:#@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
¡
Á
__inference_loss_fn_2_34167Y
>edge_to_node_kernel_regularizer_square_readvariableop_resource:ö
identity¢5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp½
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>edge_to_node_kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:ö*
dtype0¡
&Edge-to-Node/kernel/Regularizer/SquareSquare=Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö~
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
 *
×#<©
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
þ
Ò.
!__inference__traced_restore_34638
file_prefix=
"assignvariableop_e2e_conv_4_kernel:ö<
.assignvariableop_1_batch_normalization_6_gamma:;
-assignvariableop_2_batch_normalization_6_beta:B
4assignvariableop_3_batch_normalization_6_moving_mean:F
8assignvariableop_4_batch_normalization_6_moving_variance:?
$assignvariableop_5_e2e_conv_5_kernel:ö<
.assignvariableop_6_batch_normalization_7_gamma:;
-assignvariableop_7_batch_normalization_7_beta:B
4assignvariableop_8_batch_normalization_7_moving_mean:F
8assignvariableop_9_batch_normalization_7_moving_variance:B
'assignvariableop_10_edge_to_node_kernel:ö3
%assignvariableop_11_edge_to_node_bias:=
/assignvariableop_12_batch_normalization_8_gamma:<
.assignvariableop_13_batch_normalization_8_beta:C
5assignvariableop_14_batch_normalization_8_moving_mean:G
9assignvariableop_15_batch_normalization_8_moving_variance:C
(assignvariableop_16_node_to_graph_kernel:ö 4
&assignvariableop_17_node_to_graph_bias: 4
"assignvariableop_18_dense_6_kernel:#@.
 assignvariableop_19_dense_6_bias:@4
"assignvariableop_20_dense_7_kernel:@@.
 assignvariableop_21_dense_7_bias:@4
"assignvariableop_22_dense_8_kernel:@.
 assignvariableop_23_dense_8_bias:(
assignvariableop_24_nadam_iter:	 *
 assignvariableop_25_nadam_beta_1: *
 assignvariableop_26_nadam_beta_2: )
assignvariableop_27_nadam_decay: 1
'assignvariableop_28_nadam_learning_rate: 2
(assignvariableop_29_nadam_momentum_cache: #
assignvariableop_30_total: #
assignvariableop_31_count: %
assignvariableop_32_total_1: %
assignvariableop_33_count_1: -
assignvariableop_34_accumulator:H
-assignvariableop_35_nadam_e2e_conv_4_kernel_m:öE
7assignvariableop_36_nadam_batch_normalization_6_gamma_m:D
6assignvariableop_37_nadam_batch_normalization_6_beta_m:H
-assignvariableop_38_nadam_e2e_conv_5_kernel_m:öE
7assignvariableop_39_nadam_batch_normalization_7_gamma_m:D
6assignvariableop_40_nadam_batch_normalization_7_beta_m:J
/assignvariableop_41_nadam_edge_to_node_kernel_m:ö;
-assignvariableop_42_nadam_edge_to_node_bias_m:E
7assignvariableop_43_nadam_batch_normalization_8_gamma_m:D
6assignvariableop_44_nadam_batch_normalization_8_beta_m:K
0assignvariableop_45_nadam_node_to_graph_kernel_m:ö <
.assignvariableop_46_nadam_node_to_graph_bias_m: <
*assignvariableop_47_nadam_dense_6_kernel_m:#@6
(assignvariableop_48_nadam_dense_6_bias_m:@<
*assignvariableop_49_nadam_dense_7_kernel_m:@@6
(assignvariableop_50_nadam_dense_7_bias_m:@<
*assignvariableop_51_nadam_dense_8_kernel_m:@6
(assignvariableop_52_nadam_dense_8_bias_m:H
-assignvariableop_53_nadam_e2e_conv_4_kernel_v:öE
7assignvariableop_54_nadam_batch_normalization_6_gamma_v:D
6assignvariableop_55_nadam_batch_normalization_6_beta_v:H
-assignvariableop_56_nadam_e2e_conv_5_kernel_v:öE
7assignvariableop_57_nadam_batch_normalization_7_gamma_v:D
6assignvariableop_58_nadam_batch_normalization_7_beta_v:J
/assignvariableop_59_nadam_edge_to_node_kernel_v:ö;
-assignvariableop_60_nadam_edge_to_node_bias_v:E
7assignvariableop_61_nadam_batch_normalization_8_gamma_v:D
6assignvariableop_62_nadam_batch_normalization_8_beta_v:K
0assignvariableop_63_nadam_node_to_graph_kernel_v:ö <
.assignvariableop_64_nadam_node_to_graph_bias_v: <
*assignvariableop_65_nadam_dense_6_kernel_v:#@6
(assignvariableop_66_nadam_dense_6_bias_v:@<
*assignvariableop_67_nadam_dense_7_kernel_v:@@6
(assignvariableop_68_nadam_dense_7_bias_v:@<
*assignvariableop_69_nadam_dense_8_kernel_v:@6
(assignvariableop_70_nadam_dense_8_bias_v:
identity_72¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_8¢AssignVariableOp_9¼'
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*â&
valueØ&BÕ&HB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*¥
valueBHB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¶
_output_shapes£
 ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*V
dtypesL
J2H	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp"assignvariableop_e2e_conv_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp.assignvariableop_1_batch_normalization_6_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp-assignvariableop_2_batch_normalization_6_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_3AssignVariableOp4assignvariableop_3_batch_normalization_6_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_4AssignVariableOp8assignvariableop_4_batch_normalization_6_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp$assignvariableop_5_e2e_conv_5_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_7_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_7_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_8AssignVariableOp4assignvariableop_8_batch_normalization_7_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_9AssignVariableOp8assignvariableop_9_batch_normalization_7_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp'assignvariableop_10_edge_to_node_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp%assignvariableop_11_edge_to_node_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_8_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp.assignvariableop_13_batch_normalization_8_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_14AssignVariableOp5assignvariableop_14_batch_normalization_8_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_15AssignVariableOp9assignvariableop_15_batch_normalization_8_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp(assignvariableop_16_node_to_graph_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp&assignvariableop_17_node_to_graph_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_6_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_6_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_7_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_7_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_8_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp assignvariableop_23_dense_8_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_24AssignVariableOpassignvariableop_24_nadam_iterIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp assignvariableop_25_nadam_beta_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp assignvariableop_26_nadam_beta_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_nadam_decayIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp'assignvariableop_28_nadam_learning_rateIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp(assignvariableop_29_nadam_momentum_cacheIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOpassignvariableop_30_totalIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOpassignvariableop_31_countIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOpassignvariableop_32_total_1Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOpassignvariableop_33_count_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOpassignvariableop_34_accumulatorIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp-assignvariableop_35_nadam_e2e_conv_4_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_36AssignVariableOp7assignvariableop_36_nadam_batch_normalization_6_gamma_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_37AssignVariableOp6assignvariableop_37_nadam_batch_normalization_6_beta_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp-assignvariableop_38_nadam_e2e_conv_5_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_39AssignVariableOp7assignvariableop_39_nadam_batch_normalization_7_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_40AssignVariableOp6assignvariableop_40_nadam_batch_normalization_7_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_41AssignVariableOp/assignvariableop_41_nadam_edge_to_node_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp-assignvariableop_42_nadam_edge_to_node_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_43AssignVariableOp7assignvariableop_43_nadam_batch_normalization_8_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_44AssignVariableOp6assignvariableop_44_nadam_batch_normalization_8_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_45AssignVariableOp0assignvariableop_45_nadam_node_to_graph_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp.assignvariableop_46_nadam_node_to_graph_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp*assignvariableop_47_nadam_dense_6_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp(assignvariableop_48_nadam_dense_6_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp*assignvariableop_49_nadam_dense_7_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp(assignvariableop_50_nadam_dense_7_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp*assignvariableop_51_nadam_dense_8_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp(assignvariableop_52_nadam_dense_8_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOp-assignvariableop_53_nadam_e2e_conv_4_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_54AssignVariableOp7assignvariableop_54_nadam_batch_normalization_6_gamma_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_55AssignVariableOp6assignvariableop_55_nadam_batch_normalization_6_beta_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_56AssignVariableOp-assignvariableop_56_nadam_e2e_conv_5_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_57AssignVariableOp7assignvariableop_57_nadam_batch_normalization_7_gamma_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_58AssignVariableOp6assignvariableop_58_nadam_batch_normalization_7_beta_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_59AssignVariableOp/assignvariableop_59_nadam_edge_to_node_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_60AssignVariableOp-assignvariableop_60_nadam_edge_to_node_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_61AssignVariableOp7assignvariableop_61_nadam_batch_normalization_8_gamma_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_62AssignVariableOp6assignvariableop_62_nadam_batch_normalization_8_beta_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_63AssignVariableOp0assignvariableop_63_nadam_node_to_graph_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_64AssignVariableOp.assignvariableop_64_nadam_node_to_graph_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_65AssignVariableOp*assignvariableop_65_nadam_dense_6_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_66AssignVariableOp(assignvariableop_66_nadam_dense_6_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_67AssignVariableOp*assignvariableop_67_nadam_dense_7_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_68AssignVariableOp(assignvariableop_68_nadam_dense_7_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp*assignvariableop_69_nadam_dense_8_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp(assignvariableop_70_nadam_dense_8_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 é
Identity_71Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_72IdentityIdentity_71:output:0^NoOp_1*
T0*
_output_shapes
: Ö
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_72Identity_72:output:0*¥
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


ó
B__inference_dense_7_layer_call_and_return_conditional_losses_32396

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¹
r
H__inference_concatenate_2_layer_call_and_return_conditional_losses_32359

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
:ÿÿÿÿÿÿÿÿÿ#W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ó
B__inference_dense_6_layer_call_and_return_conditional_losses_32372

inputs0
matmul_readvariableop_resource:#@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:#@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
Ù
¿
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_33937

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®

+__inference_BrainNetCNN_layer_call_fn_32904
	input_img
input_struc"
unknown:ö
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:$
	unknown_4:ö
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:$
	unknown_9:ö

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:%

unknown_15:ö 

unknown_16: 

unknown_17:#@

unknown_18:@

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	input_imginput_strucunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_BrainNetCNN_layer_call_and_return_conditional_losses_32799o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿöö:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
#
_user_specified_name	input_img:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_struc
÷
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_32342

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¨

+__inference_BrainNetCNN_layer_call_fn_33172
inputs_0
inputs_1"
unknown:ö
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:$
	unknown_4:ö
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:$
	unknown_9:ö

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:%

unknown_15:ö 

unknown_16: 

unknown_17:#@

unknown_18:@

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_BrainNetCNN_layer_call_and_return_conditional_losses_32451o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿöö:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Á

'__inference_dense_8_layer_call_fn_34123

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallÚ
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
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_32420o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Á
t
H__inference_concatenate_2_layer_call_and_return_conditional_losses_34020
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
:ÿÿÿÿÿÿÿÿÿ#W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ù
¿
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_33843

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_32350

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
´

+__inference_BrainNetCNN_layer_call_fn_32502
	input_img
input_struc"
unknown:ö
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:$
	unknown_4:ö
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:$
	unknown_9:ö

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:%

unknown_15:ö 

unknown_16: 

unknown_17:#@

unknown_18:@

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	input_imginput_strucunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_BrainNetCNN_layer_call_and_return_conditional_losses_32451o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿöö:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
#
_user_specified_name	input_img:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_struc
{
ò
E__inference_e2e_conv_4_layer_call_and_return_conditional_losses_32222

inputs2
readvariableop_resource:ö
identity¢ReadVariableOp¢ReadVariableOp_1¢3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOpo
ReadVariableOpReadVariableOpreadvariableop_resource*'
_output_shapes
:ö*
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
valueB"      
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   ö         t
ReshapeReshapestrided_slice:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:öq
ReadVariableOp_1ReadVariableOpreadvariableop_resource*'
_output_shapes
:ö*
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
valueB"      
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_maskh
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ö            z
	Reshape_1Reshapestrided_slice_1:output:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:ö
convolutionConv2DinputsReshape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides

convolution_1Conv2DinputsReshape_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :þ.
concatConcatV2convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0concat/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿööO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :+
concat_1ConcatV2convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0concat_1/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿööl
addAddV2concat:output:0concat_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*'
_output_shapes
:ö*
dtype0
$e2e_conv_4/kernel/Regularizer/SquareSquare;e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö|
#e2e_conv_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¡
!e2e_conv_4/kernel/Regularizer/SumSum(e2e_conv_4/kernel/Regularizer/Square:y:0,e2e_conv_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#e2e_conv_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<£
!e2e_conv_4/kernel/Regularizer/mulMul,e2e_conv_4/kernel/Regularizer/mul/x:output:0*e2e_conv_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö 
NoOpNoOp^ReadVariableOp^ReadVariableOp_14^e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿöö: 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12j
3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
 
_user_specified_nameinputs
À
E
)__inference_dropout_6_layer_call_fn_33974

inputs
identityº
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_32342h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

b
)__inference_dropout_6_layer_call_fn_33979

inputs
identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_32611w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
­
Y
-__inference_concatenate_2_layer_call_fn_34013
inputs_0
inputs_1
identityÃ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_32359`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ïl
¡
F__inference_BrainNetCNN_layer_call_and_return_conditional_losses_32996
	input_img
input_struc+
e2e_conv_4_32908:ö)
batch_normalization_6_32911:)
batch_normalization_6_32913:)
batch_normalization_6_32915:)
batch_normalization_6_32917:+
e2e_conv_5_32920:ö)
batch_normalization_7_32923:)
batch_normalization_7_32925:)
batch_normalization_7_32927:)
batch_normalization_7_32929:-
edge_to_node_32932:ö 
edge_to_node_32934:)
batch_normalization_8_32937:)
batch_normalization_8_32939:)
batch_normalization_8_32941:)
batch_normalization_8_32943:.
node_to_graph_32946:ö !
node_to_graph_32948: 
dense_6_32954:#@
dense_6_32956:@
dense_7_32960:@@
dense_7_32962:@
dense_8_32966:@
dense_8_32968:
identity¢$Edge-to-Node/StatefulPartitionedCall¢5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp¢%Node-to-Graph/StatefulPartitionedCall¢6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢"e2e_conv_4/StatefulPartitionedCall¢3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp¢"e2e_conv_5/StatefulPartitionedCall¢3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOpñ
"e2e_conv_4/StatefulPartitionedCallStatefulPartitionedCall	input_imge2e_conv_4_32908*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_e2e_conv_4_layer_call_and_return_conditional_losses_32222
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall+e2e_conv_4/StatefulPartitionedCall:output:0batch_normalization_6_32911batch_normalization_6_32913batch_normalization_6_32915batch_normalization_6_32917*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_32009
"e2e_conv_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0e2e_conv_5_32920*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_e2e_conv_5_layer_call_and_return_conditional_losses_32269
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall+e2e_conv_5/StatefulPartitionedCall:output:0batch_normalization_7_32923batch_normalization_7_32925batch_normalization_7_32927batch_normalization_7_32929*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_32073¹
$Edge-to-Node/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0edge_to_node_32932edge_to_node_32934*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_Edge-to-Node_layer_call_and_return_conditional_losses_32299
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall-Edge-to-Node/StatefulPartitionedCall:output:0batch_normalization_8_32937batch_normalization_8_32939batch_normalization_8_32941batch_normalization_8_32943*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_32137¼
%Node-to-Graph/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0node_to_graph_32946node_to_graph_32948*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Node-to-Graph_layer_call_and_return_conditional_losses_32331ì
dropout_6/PartitionedCallPartitionedCall.Node-to-Graph/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_32342Ø
flatten_2/PartitionedCallPartitionedCall"dropout_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_32350î
concatenate_2/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0input_struc*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_32359
dense_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_6_32954dense_6_32956*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_32372Þ
dropout_7/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_32383
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_7_32960dense_7_32962*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_32396Þ
dropout_8/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_32407
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_8_32966dense_8_32968*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_32420
3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpe2e_conv_4_32908*'
_output_shapes
:ö*
dtype0
$e2e_conv_4/kernel/Regularizer/SquareSquare;e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö|
#e2e_conv_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¡
!e2e_conv_4/kernel/Regularizer/SumSum(e2e_conv_4/kernel/Regularizer/Square:y:0,e2e_conv_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#e2e_conv_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<£
!e2e_conv_4/kernel/Regularizer/mulMul,e2e_conv_4/kernel/Regularizer/mul/x:output:0*e2e_conv_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpe2e_conv_5_32920*'
_output_shapes
:ö*
dtype0
$e2e_conv_5/kernel/Regularizer/SquareSquare;e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö|
#e2e_conv_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¡
!e2e_conv_5/kernel/Regularizer/SumSum(e2e_conv_5/kernel/Regularizer/Square:y:0,e2e_conv_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#e2e_conv_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<£
!e2e_conv_5/kernel/Regularizer/mulMul,e2e_conv_5/kernel/Regularizer/mul/x:output:0*e2e_conv_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOpReadVariableOpedge_to_node_32932*'
_output_shapes
:ö*
dtype0¡
&Edge-to-Node/kernel/Regularizer/SquareSquare=Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö~
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
 *
×#<©
#Edge-to-Node/kernel/Regularizer/mulMul.Edge-to-Node/kernel/Regularizer/mul/x:output:0,Edge-to-Node/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOpReadVariableOpnode_to_graph_32946*'
_output_shapes
:ö *
dtype0£
'Node-to-Graph/kernel/Regularizer/SquareSquare>Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö 
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
 *
×#<¬
$Node-to-Graph/kernel/Regularizer/mulMul/Node-to-Graph/kernel/Regularizer/mul/x:output:0-Node-to-Graph/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp%^Edge-to-Node/StatefulPartitionedCall6^Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp&^Node-to-Graph/StatefulPartitionedCall7^Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall#^e2e_conv_4/StatefulPartitionedCall4^e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp#^e2e_conv_5/StatefulPartitionedCall4^e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿöö:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 2L
$Edge-to-Node/StatefulPartitionedCall$Edge-to-Node/StatefulPartitionedCall2n
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp2N
%Node-to-Graph/StatefulPartitionedCall%Node-to-Graph/StatefulPartitionedCall2p
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2H
"e2e_conv_4/StatefulPartitionedCall"e2e_conv_4/StatefulPartitionedCall2j
3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp2H
"e2e_conv_5/StatefulPartitionedCall"e2e_conv_5/StatefulPartitionedCall2j
3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp:\ X
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
#
_user_specified_name	input_img:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_struc
q

F__inference_BrainNetCNN_layer_call_and_return_conditional_losses_33088
	input_img
input_struc+
e2e_conv_4_33000:ö)
batch_normalization_6_33003:)
batch_normalization_6_33005:)
batch_normalization_6_33007:)
batch_normalization_6_33009:+
e2e_conv_5_33012:ö)
batch_normalization_7_33015:)
batch_normalization_7_33017:)
batch_normalization_7_33019:)
batch_normalization_7_33021:-
edge_to_node_33024:ö 
edge_to_node_33026:)
batch_normalization_8_33029:)
batch_normalization_8_33031:)
batch_normalization_8_33033:)
batch_normalization_8_33035:.
node_to_graph_33038:ö !
node_to_graph_33040: 
dense_6_33046:#@
dense_6_33048:@
dense_7_33052:@@
dense_7_33054:@
dense_8_33058:@
dense_8_33060:
identity¢$Edge-to-Node/StatefulPartitionedCall¢5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp¢%Node-to-Graph/StatefulPartitionedCall¢6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢!dropout_6/StatefulPartitionedCall¢!dropout_7/StatefulPartitionedCall¢!dropout_8/StatefulPartitionedCall¢"e2e_conv_4/StatefulPartitionedCall¢3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp¢"e2e_conv_5/StatefulPartitionedCall¢3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOpñ
"e2e_conv_4/StatefulPartitionedCallStatefulPartitionedCall	input_imge2e_conv_4_33000*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_e2e_conv_4_layer_call_and_return_conditional_losses_32222
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall+e2e_conv_4/StatefulPartitionedCall:output:0batch_normalization_6_33003batch_normalization_6_33005batch_normalization_6_33007batch_normalization_6_33009*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_32040
"e2e_conv_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0e2e_conv_5_33012*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_e2e_conv_5_layer_call_and_return_conditional_losses_32269
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall+e2e_conv_5/StatefulPartitionedCall:output:0batch_normalization_7_33015batch_normalization_7_33017batch_normalization_7_33019batch_normalization_7_33021*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_32104¹
$Edge-to-Node/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0edge_to_node_33024edge_to_node_33026*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_Edge-to-Node_layer_call_and_return_conditional_losses_32299
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall-Edge-to-Node/StatefulPartitionedCall:output:0batch_normalization_8_33029batch_normalization_8_33031batch_normalization_8_33033batch_normalization_8_33035*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_32168¼
%Node-to-Graph/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0node_to_graph_33038node_to_graph_33040*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Node-to-Graph_layer_call_and_return_conditional_losses_32331ü
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall.Node-to-Graph/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_32611à
flatten_2/PartitionedCallPartitionedCall*dropout_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_32350î
concatenate_2/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0input_struc*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_32359
dense_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_6_33046dense_6_33048*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_32372
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_32565
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_7_33052dense_7_33054*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_32396
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_32532
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_8_33058dense_8_33060*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_32420
3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpe2e_conv_4_33000*'
_output_shapes
:ö*
dtype0
$e2e_conv_4/kernel/Regularizer/SquareSquare;e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö|
#e2e_conv_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¡
!e2e_conv_4/kernel/Regularizer/SumSum(e2e_conv_4/kernel/Regularizer/Square:y:0,e2e_conv_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#e2e_conv_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<£
!e2e_conv_4/kernel/Regularizer/mulMul,e2e_conv_4/kernel/Regularizer/mul/x:output:0*e2e_conv_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpe2e_conv_5_33012*'
_output_shapes
:ö*
dtype0
$e2e_conv_5/kernel/Regularizer/SquareSquare;e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö|
#e2e_conv_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¡
!e2e_conv_5/kernel/Regularizer/SumSum(e2e_conv_5/kernel/Regularizer/Square:y:0,e2e_conv_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#e2e_conv_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<£
!e2e_conv_5/kernel/Regularizer/mulMul,e2e_conv_5/kernel/Regularizer/mul/x:output:0*e2e_conv_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOpReadVariableOpedge_to_node_33024*'
_output_shapes
:ö*
dtype0¡
&Edge-to-Node/kernel/Regularizer/SquareSquare=Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö~
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
 *
×#<©
#Edge-to-Node/kernel/Regularizer/mulMul.Edge-to-Node/kernel/Regularizer/mul/x:output:0,Edge-to-Node/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOpReadVariableOpnode_to_graph_33038*'
_output_shapes
:ö *
dtype0£
'Node-to-Graph/kernel/Regularizer/SquareSquare>Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö 
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
 *
×#<¬
$Node-to-Graph/kernel/Regularizer/mulMul/Node-to-Graph/kernel/Regularizer/mul/x:output:0-Node-to-Graph/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp%^Edge-to-Node/StatefulPartitionedCall6^Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp&^Node-to-Graph/StatefulPartitionedCall7^Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall#^e2e_conv_4/StatefulPartitionedCall4^e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp#^e2e_conv_5/StatefulPartitionedCall4^e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿöö:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 2L
$Edge-to-Node/StatefulPartitionedCall$Edge-to-Node/StatefulPartitionedCall2n
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp2N
%Node-to-Graph/StatefulPartitionedCall%Node-to-Graph/StatefulPartitionedCall2p
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2H
"e2e_conv_4/StatefulPartitionedCall"e2e_conv_4/StatefulPartitionedCall2j
3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp2H
"e2e_conv_5/StatefulPartitionedCall"e2e_conv_5/StatefulPartitionedCall2j
3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp:\ X
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
#
_user_specified_name	input_img:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_struc
 
E
)__inference_dropout_7_layer_call_fn_34045

inputs
identity²
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
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_32383`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
×
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_34055

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
þ
 
__inference__traced_save_34415
file_prefix0
,savev2_e2e_conv_4_kernel_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop0
,savev2_e2e_conv_5_kernel_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop2
.savev2_edge_to_node_kernel_read_readvariableop0
,savev2_edge_to_node_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop3
/savev2_node_to_graph_kernel_read_readvariableop1
-savev2_node_to_graph_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop*
&savev2_accumulator_read_readvariableop8
4savev2_nadam_e2e_conv_4_kernel_m_read_readvariableopB
>savev2_nadam_batch_normalization_6_gamma_m_read_readvariableopA
=savev2_nadam_batch_normalization_6_beta_m_read_readvariableop8
4savev2_nadam_e2e_conv_5_kernel_m_read_readvariableopB
>savev2_nadam_batch_normalization_7_gamma_m_read_readvariableopA
=savev2_nadam_batch_normalization_7_beta_m_read_readvariableop:
6savev2_nadam_edge_to_node_kernel_m_read_readvariableop8
4savev2_nadam_edge_to_node_bias_m_read_readvariableopB
>savev2_nadam_batch_normalization_8_gamma_m_read_readvariableopA
=savev2_nadam_batch_normalization_8_beta_m_read_readvariableop;
7savev2_nadam_node_to_graph_kernel_m_read_readvariableop9
5savev2_nadam_node_to_graph_bias_m_read_readvariableop5
1savev2_nadam_dense_6_kernel_m_read_readvariableop3
/savev2_nadam_dense_6_bias_m_read_readvariableop5
1savev2_nadam_dense_7_kernel_m_read_readvariableop3
/savev2_nadam_dense_7_bias_m_read_readvariableop5
1savev2_nadam_dense_8_kernel_m_read_readvariableop3
/savev2_nadam_dense_8_bias_m_read_readvariableop8
4savev2_nadam_e2e_conv_4_kernel_v_read_readvariableopB
>savev2_nadam_batch_normalization_6_gamma_v_read_readvariableopA
=savev2_nadam_batch_normalization_6_beta_v_read_readvariableop8
4savev2_nadam_e2e_conv_5_kernel_v_read_readvariableopB
>savev2_nadam_batch_normalization_7_gamma_v_read_readvariableopA
=savev2_nadam_batch_normalization_7_beta_v_read_readvariableop:
6savev2_nadam_edge_to_node_kernel_v_read_readvariableop8
4savev2_nadam_edge_to_node_bias_v_read_readvariableopB
>savev2_nadam_batch_normalization_8_gamma_v_read_readvariableopA
=savev2_nadam_batch_normalization_8_beta_v_read_readvariableop;
7savev2_nadam_node_to_graph_kernel_v_read_readvariableop9
5savev2_nadam_node_to_graph_bias_v_read_readvariableop5
1savev2_nadam_dense_6_kernel_v_read_readvariableop3
/savev2_nadam_dense_6_bias_v_read_readvariableop5
1savev2_nadam_dense_7_kernel_v_read_readvariableop3
/savev2_nadam_dense_7_bias_v_read_readvariableop5
1savev2_nadam_dense_8_kernel_v_read_readvariableop3
/savev2_nadam_dense_8_bias_v_read_readvariableop
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
: ¹'
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*â&
valueØ&BÕ&HB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*¥
valueBHB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ö
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_e2e_conv_4_kernel_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop,savev2_e2e_conv_5_kernel_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop.savev2_edge_to_node_kernel_read_readvariableop,savev2_edge_to_node_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop/savev2_node_to_graph_kernel_read_readvariableop-savev2_node_to_graph_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop&savev2_accumulator_read_readvariableop4savev2_nadam_e2e_conv_4_kernel_m_read_readvariableop>savev2_nadam_batch_normalization_6_gamma_m_read_readvariableop=savev2_nadam_batch_normalization_6_beta_m_read_readvariableop4savev2_nadam_e2e_conv_5_kernel_m_read_readvariableop>savev2_nadam_batch_normalization_7_gamma_m_read_readvariableop=savev2_nadam_batch_normalization_7_beta_m_read_readvariableop6savev2_nadam_edge_to_node_kernel_m_read_readvariableop4savev2_nadam_edge_to_node_bias_m_read_readvariableop>savev2_nadam_batch_normalization_8_gamma_m_read_readvariableop=savev2_nadam_batch_normalization_8_beta_m_read_readvariableop7savev2_nadam_node_to_graph_kernel_m_read_readvariableop5savev2_nadam_node_to_graph_bias_m_read_readvariableop1savev2_nadam_dense_6_kernel_m_read_readvariableop/savev2_nadam_dense_6_bias_m_read_readvariableop1savev2_nadam_dense_7_kernel_m_read_readvariableop/savev2_nadam_dense_7_bias_m_read_readvariableop1savev2_nadam_dense_8_kernel_m_read_readvariableop/savev2_nadam_dense_8_bias_m_read_readvariableop4savev2_nadam_e2e_conv_4_kernel_v_read_readvariableop>savev2_nadam_batch_normalization_6_gamma_v_read_readvariableop=savev2_nadam_batch_normalization_6_beta_v_read_readvariableop4savev2_nadam_e2e_conv_5_kernel_v_read_readvariableop>savev2_nadam_batch_normalization_7_gamma_v_read_readvariableop=savev2_nadam_batch_normalization_7_beta_v_read_readvariableop6savev2_nadam_edge_to_node_kernel_v_read_readvariableop4savev2_nadam_edge_to_node_bias_v_read_readvariableop>savev2_nadam_batch_normalization_8_gamma_v_read_readvariableop=savev2_nadam_batch_normalization_8_beta_v_read_readvariableop7savev2_nadam_node_to_graph_kernel_v_read_readvariableop5savev2_nadam_node_to_graph_bias_v_read_readvariableop1savev2_nadam_dense_6_kernel_v_read_readvariableop/savev2_nadam_dense_6_bias_v_read_readvariableop1savev2_nadam_dense_7_kernel_v_read_readvariableop/savev2_nadam_dense_7_bias_v_read_readvariableop1savev2_nadam_dense_8_kernel_v_read_readvariableop/savev2_nadam_dense_8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *V
dtypesL
J2H	
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

identity_1Identity_1:output:0*Û
_input_shapesÉ
Æ: :ö:::::ö:::::ö::::::ö : :#@:@:@@:@:@:: : : : : : : : : : ::ö:::ö:::ö::::ö : :#@:@:@@:@:@::ö:::ö:::ö::::ö : :#@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:ö: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::-)
'
_output_shapes
:ö: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
::-)
'
_output_shapes
:ö: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::-)
'
_output_shapes
:ö : 

_output_shapes
: :$ 

_output_shapes

:#@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: : #

_output_shapes
::-$)
'
_output_shapes
:ö: %

_output_shapes
:: &

_output_shapes
::-')
'
_output_shapes
:ö: (

_output_shapes
:: )

_output_shapes
::-*)
'
_output_shapes
:ö: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
::-.)
'
_output_shapes
:ö : /

_output_shapes
: :$0 

_output_shapes

:#@: 1

_output_shapes
:@:$2 

_output_shapes

:@@: 3

_output_shapes
:@:$4 

_output_shapes

:@: 5

_output_shapes
::-6)
'
_output_shapes
:ö: 7

_output_shapes
:: 8

_output_shapes
::-9)
'
_output_shapes
:ö: :

_output_shapes
:: ;

_output_shapes
::-<)
'
_output_shapes
:ö: =

_output_shapes
:: >

_output_shapes
:: ?

_output_shapes
::-@)
'
_output_shapes
:ö : A

_output_shapes
: :$B 

_output_shapes

:#@: C

_output_shapes
:@:$D 

_output_shapes

:@@: E

_output_shapes
:@:$F 

_output_shapes

:@: G

_output_shapes
::H

_output_shapes
: 

»
H__inference_Node-to-Graph_layer_call_and_return_conditional_losses_33969

inputs9
conv2d_readvariableop_resource:ö -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:ö *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:ö *
dtype0£
'Node-to-Graph/kernel/Regularizer/SquareSquare>Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö 
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
 *
×#<¬
$Node-to-Graph/kernel/Regularizer/mulMul/Node-to-Graph/kernel/Regularizer/mul/x:output:0-Node-to-Graph/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp7^Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿö: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2p
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
 
_user_specified_nameinputs
Ë

P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_33919

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ð
5__inference_batch_normalization_8_layer_call_fn_33901

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_32168
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
{
ò
E__inference_e2e_conv_5_layer_call_and_return_conditional_losses_33781

inputs2
readvariableop_resource:ö
identity¢ReadVariableOp¢ReadVariableOp_1¢3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOpo
ReadVariableOpReadVariableOpreadvariableop_resource*'
_output_shapes
:ö*
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
valueB"      
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   ö         t
ReshapeReshapestrided_slice:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:öq
ReadVariableOp_1ReadVariableOpreadvariableop_resource*'
_output_shapes
:ö*
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
valueB"      
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_maskh
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ö            z
	Reshape_1Reshapestrided_slice_1:output:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:ö
convolutionConv2DinputsReshape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides

convolution_1Conv2DinputsReshape_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :þ.
concatConcatV2convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0concat/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿööO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :+
concat_1ConcatV2convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0concat_1/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿööl
addAddV2concat:output:0concat_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*'
_output_shapes
:ö*
dtype0
$e2e_conv_5/kernel/Regularizer/SquareSquare;e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö|
#e2e_conv_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¡
!e2e_conv_5/kernel/Regularizer/SumSum(e2e_conv_5/kernel/Regularizer/Square:y:0,e2e_conv_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#e2e_conv_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<£
!e2e_conv_5/kernel/Regularizer/mulMul,e2e_conv_5/kernel/Regularizer/mul/x:output:0*e2e_conv_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö 
NoOpNoOp^ReadVariableOp^ReadVariableOp_14^e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿöö: 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12j
3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
 
_user_specified_nameinputs
ò	
c
D__inference_dropout_8_layer_call_and_return_conditional_losses_32532

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
×
b
D__inference_dropout_8_layer_call_and_return_conditional_losses_34102

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ð

*__inference_e2e_conv_4_layer_call_fn_33638

inputs"
unknown:ö
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_e2e_conv_4_layer_call_and_return_conditional_losses_32222y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿöö: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
 
_user_specified_nameinputs
ò	
c
D__inference_dropout_8_layer_call_and_return_conditional_losses_34114

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


ó
B__inference_dense_8_layer_call_and_return_conditional_losses_34134

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
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
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ø
£
-__inference_Node-to-Graph_layer_call_fn_33952

inputs"
unknown:ö 
	unknown_0: 
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Node-to-Graph_layer_call_and_return_conditional_losses_32331w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿö: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
 
_user_specified_nameinputs
Ë

P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_33825

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ó
B__inference_dense_7_layer_call_and_return_conditional_losses_34087

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


ó
B__inference_dense_8_layer_call_and_return_conditional_losses_32420

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
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
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

»
H__inference_Node-to-Graph_layer_call_and_return_conditional_losses_32331

inputs9
conv2d_readvariableop_resource:ö -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:ö *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:ö *
dtype0£
'Node-to-Graph/kernel/Regularizer/SquareSquare>Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö 
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
 *
×#<¬
$Node-to-Graph/kernel/Regularizer/mulMul/Node-to-Graph/kernel/Regularizer/mul/x:output:0-Node-to-Graph/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp7^Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿö: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2p
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
 
_user_specified_nameinputs
×
b
D__inference_dropout_8_layer_call_and_return_conditional_losses_32407

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ø
¹
G__inference_Edge-to-Node_layer_call_and_return_conditional_losses_32299

inputs9
conv2d_readvariableop_resource:ö-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:ö*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿöY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:ö*
dtype0¡
&Edge-to-Node/kernel/Regularizer/SquareSquare=Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö~
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
 *
×#<©
#Edge-to-Node/kernel/Regularizer/mulMul.Edge-to-Node/kernel/Regularizer/mul/x:output:0,Edge-to-Node/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö¯
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp6^Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿöö: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2n
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
 
_user_specified_nameinputs
ý
½
__inference_loss_fn_0_34145W
<e2e_conv_4_kernel_regularizer_square_readvariableop_resource:ö
identity¢3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp¹
3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<e2e_conv_4_kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:ö*
dtype0
$e2e_conv_4/kernel/Regularizer/SquareSquare;e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö|
#e2e_conv_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¡
!e2e_conv_4/kernel/Regularizer/SumSum(e2e_conv_4/kernel/Regularizer/Square:y:0,e2e_conv_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#e2e_conv_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<£
!e2e_conv_4/kernel/Regularizer/mulMul,e2e_conv_4/kernel/Regularizer/mul/x:output:0*e2e_conv_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
IdentityIdentity%e2e_conv_4/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp
÷
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_33984

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ò
b
)__inference_dropout_7_layer_call_fn_34050

inputs
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_32565o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 
E
)__inference_dropout_8_layer_call_fn_34092

inputs
identity²
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
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_32407`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ü
é
 __inference__wrapped_model_31987
	input_img
input_strucI
.brainnetcnn_e2e_conv_4_readvariableop_resource:öG
9brainnetcnn_batch_normalization_6_readvariableop_resource:I
;brainnetcnn_batch_normalization_6_readvariableop_1_resource:X
Jbrainnetcnn_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:Z
Lbrainnetcnn_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:I
.brainnetcnn_e2e_conv_5_readvariableop_resource:öG
9brainnetcnn_batch_normalization_7_readvariableop_resource:I
;brainnetcnn_batch_normalization_7_readvariableop_1_resource:X
Jbrainnetcnn_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:Z
Lbrainnetcnn_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:R
7brainnetcnn_edge_to_node_conv2d_readvariableop_resource:öF
8brainnetcnn_edge_to_node_biasadd_readvariableop_resource:G
9brainnetcnn_batch_normalization_8_readvariableop_resource:I
;brainnetcnn_batch_normalization_8_readvariableop_1_resource:X
Jbrainnetcnn_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:Z
Lbrainnetcnn_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:S
8brainnetcnn_node_to_graph_conv2d_readvariableop_resource:ö G
9brainnetcnn_node_to_graph_biasadd_readvariableop_resource: D
2brainnetcnn_dense_6_matmul_readvariableop_resource:#@A
3brainnetcnn_dense_6_biasadd_readvariableop_resource:@D
2brainnetcnn_dense_7_matmul_readvariableop_resource:@@A
3brainnetcnn_dense_7_biasadd_readvariableop_resource:@D
2brainnetcnn_dense_8_matmul_readvariableop_resource:@A
3brainnetcnn_dense_8_biasadd_readvariableop_resource:
identity¢/BrainNetCNN/Edge-to-Node/BiasAdd/ReadVariableOp¢.BrainNetCNN/Edge-to-Node/Conv2D/ReadVariableOp¢0BrainNetCNN/Node-to-Graph/BiasAdd/ReadVariableOp¢/BrainNetCNN/Node-to-Graph/Conv2D/ReadVariableOp¢ABrainNetCNN/batch_normalization_6/FusedBatchNormV3/ReadVariableOp¢CBrainNetCNN/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1¢0BrainNetCNN/batch_normalization_6/ReadVariableOp¢2BrainNetCNN/batch_normalization_6/ReadVariableOp_1¢ABrainNetCNN/batch_normalization_7/FusedBatchNormV3/ReadVariableOp¢CBrainNetCNN/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1¢0BrainNetCNN/batch_normalization_7/ReadVariableOp¢2BrainNetCNN/batch_normalization_7/ReadVariableOp_1¢ABrainNetCNN/batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢CBrainNetCNN/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢0BrainNetCNN/batch_normalization_8/ReadVariableOp¢2BrainNetCNN/batch_normalization_8/ReadVariableOp_1¢*BrainNetCNN/dense_6/BiasAdd/ReadVariableOp¢)BrainNetCNN/dense_6/MatMul/ReadVariableOp¢*BrainNetCNN/dense_7/BiasAdd/ReadVariableOp¢)BrainNetCNN/dense_7/MatMul/ReadVariableOp¢*BrainNetCNN/dense_8/BiasAdd/ReadVariableOp¢)BrainNetCNN/dense_8/MatMul/ReadVariableOp¢%BrainNetCNN/e2e_conv_4/ReadVariableOp¢'BrainNetCNN/e2e_conv_4/ReadVariableOp_1¢%BrainNetCNN/e2e_conv_5/ReadVariableOp¢'BrainNetCNN/e2e_conv_5/ReadVariableOp_1
%BrainNetCNN/e2e_conv_4/ReadVariableOpReadVariableOp.brainnetcnn_e2e_conv_4_readvariableop_resource*'
_output_shapes
:ö*
dtype0{
*BrainNetCNN/e2e_conv_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,BrainNetCNN/e2e_conv_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,BrainNetCNN/e2e_conv_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      û
$BrainNetCNN/e2e_conv_4/strided_sliceStridedSlice-BrainNetCNN/e2e_conv_4/ReadVariableOp:value:03BrainNetCNN/e2e_conv_4/strided_slice/stack:output:05BrainNetCNN/e2e_conv_4/strided_slice/stack_1:output:05BrainNetCNN/e2e_conv_4/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_mask}
$BrainNetCNN/e2e_conv_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   ö         ¹
BrainNetCNN/e2e_conv_4/ReshapeReshape-BrainNetCNN/e2e_conv_4/strided_slice:output:0-BrainNetCNN/e2e_conv_4/Reshape/shape:output:0*
T0*'
_output_shapes
:ö
'BrainNetCNN/e2e_conv_4/ReadVariableOp_1ReadVariableOp.brainnetcnn_e2e_conv_4_readvariableop_resource*'
_output_shapes
:ö*
dtype0}
,BrainNetCNN/e2e_conv_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.BrainNetCNN/e2e_conv_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.BrainNetCNN/e2e_conv_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
&BrainNetCNN/e2e_conv_4/strided_slice_1StridedSlice/BrainNetCNN/e2e_conv_4/ReadVariableOp_1:value:05BrainNetCNN/e2e_conv_4/strided_slice_1/stack:output:07BrainNetCNN/e2e_conv_4/strided_slice_1/stack_1:output:07BrainNetCNN/e2e_conv_4/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_mask
&BrainNetCNN/e2e_conv_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ö            ¿
 BrainNetCNN/e2e_conv_4/Reshape_1Reshape/BrainNetCNN/e2e_conv_4/strided_slice_1:output:0/BrainNetCNN/e2e_conv_4/Reshape_1/shape:output:0*
T0*'
_output_shapes
:öÄ
"BrainNetCNN/e2e_conv_4/convolutionConv2D	input_img'BrainNetCNN/e2e_conv_4/Reshape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides
È
$BrainNetCNN/e2e_conv_4/convolution_1Conv2D	input_img)BrainNetCNN/e2e_conv_4/Reshape_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides
d
"BrainNetCNN/e2e_conv_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Æ[
BrainNetCNN/e2e_conv_4/concatConcatV2-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0-BrainNetCNN/e2e_conv_4/convolution_1:output:0+BrainNetCNN/e2e_conv_4/concat/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿööf
$BrainNetCNN/e2e_conv_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :ÞW
BrainNetCNN/e2e_conv_4/concat_1ConcatV2+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0+BrainNetCNN/e2e_conv_4/convolution:output:0-BrainNetCNN/e2e_conv_4/concat_1/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö±
BrainNetCNN/e2e_conv_4/addAddV2&BrainNetCNN/e2e_conv_4/concat:output:0(BrainNetCNN/e2e_conv_4/concat_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö¦
0BrainNetCNN/batch_normalization_6/ReadVariableOpReadVariableOp9brainnetcnn_batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype0ª
2BrainNetCNN/batch_normalization_6/ReadVariableOp_1ReadVariableOp;brainnetcnn_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype0È
ABrainNetCNN/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJbrainnetcnn_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ì
CBrainNetCNN/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLbrainnetcnn_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ú
2BrainNetCNN/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3BrainNetCNN/e2e_conv_4/add:z:08BrainNetCNN/batch_normalization_6/ReadVariableOp:value:0:BrainNetCNN/batch_normalization_6/ReadVariableOp_1:value:0IBrainNetCNN/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0KBrainNetCNN/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿöö:::::*
epsilon%o:*
is_training( 
%BrainNetCNN/e2e_conv_5/ReadVariableOpReadVariableOp.brainnetcnn_e2e_conv_5_readvariableop_resource*'
_output_shapes
:ö*
dtype0{
*BrainNetCNN/e2e_conv_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,BrainNetCNN/e2e_conv_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,BrainNetCNN/e2e_conv_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      û
$BrainNetCNN/e2e_conv_5/strided_sliceStridedSlice-BrainNetCNN/e2e_conv_5/ReadVariableOp:value:03BrainNetCNN/e2e_conv_5/strided_slice/stack:output:05BrainNetCNN/e2e_conv_5/strided_slice/stack_1:output:05BrainNetCNN/e2e_conv_5/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_mask}
$BrainNetCNN/e2e_conv_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   ö         ¹
BrainNetCNN/e2e_conv_5/ReshapeReshape-BrainNetCNN/e2e_conv_5/strided_slice:output:0-BrainNetCNN/e2e_conv_5/Reshape/shape:output:0*
T0*'
_output_shapes
:ö
'BrainNetCNN/e2e_conv_5/ReadVariableOp_1ReadVariableOp.brainnetcnn_e2e_conv_5_readvariableop_resource*'
_output_shapes
:ö*
dtype0}
,BrainNetCNN/e2e_conv_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.BrainNetCNN/e2e_conv_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.BrainNetCNN/e2e_conv_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
&BrainNetCNN/e2e_conv_5/strided_slice_1StridedSlice/BrainNetCNN/e2e_conv_5/ReadVariableOp_1:value:05BrainNetCNN/e2e_conv_5/strided_slice_1/stack:output:07BrainNetCNN/e2e_conv_5/strided_slice_1/stack_1:output:07BrainNetCNN/e2e_conv_5/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_mask
&BrainNetCNN/e2e_conv_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ö            ¿
 BrainNetCNN/e2e_conv_5/Reshape_1Reshape/BrainNetCNN/e2e_conv_5/strided_slice_1:output:0/BrainNetCNN/e2e_conv_5/Reshape_1/shape:output:0*
T0*'
_output_shapes
:öñ
"BrainNetCNN/e2e_conv_5/convolutionConv2D6BrainNetCNN/batch_normalization_6/FusedBatchNormV3:y:0'BrainNetCNN/e2e_conv_5/Reshape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides
õ
$BrainNetCNN/e2e_conv_5/convolution_1Conv2D6BrainNetCNN/batch_normalization_6/FusedBatchNormV3:y:0)BrainNetCNN/e2e_conv_5/Reshape_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides
d
"BrainNetCNN/e2e_conv_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Æ[
BrainNetCNN/e2e_conv_5/concatConcatV2-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0-BrainNetCNN/e2e_conv_5/convolution_1:output:0+BrainNetCNN/e2e_conv_5/concat/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿööf
$BrainNetCNN/e2e_conv_5/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :ÞW
BrainNetCNN/e2e_conv_5/concat_1ConcatV2+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0+BrainNetCNN/e2e_conv_5/convolution:output:0-BrainNetCNN/e2e_conv_5/concat_1/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö±
BrainNetCNN/e2e_conv_5/addAddV2&BrainNetCNN/e2e_conv_5/concat:output:0(BrainNetCNN/e2e_conv_5/concat_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö¦
0BrainNetCNN/batch_normalization_7/ReadVariableOpReadVariableOp9brainnetcnn_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype0ª
2BrainNetCNN/batch_normalization_7/ReadVariableOp_1ReadVariableOp;brainnetcnn_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype0È
ABrainNetCNN/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJbrainnetcnn_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ì
CBrainNetCNN/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLbrainnetcnn_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ú
2BrainNetCNN/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3BrainNetCNN/e2e_conv_5/add:z:08BrainNetCNN/batch_normalization_7/ReadVariableOp:value:0:BrainNetCNN/batch_normalization_7/ReadVariableOp_1:value:0IBrainNetCNN/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0KBrainNetCNN/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿöö:::::*
epsilon%o:*
is_training( ¯
.BrainNetCNN/Edge-to-Node/Conv2D/ReadVariableOpReadVariableOp7brainnetcnn_edge_to_node_conv2d_readvariableop_resource*'
_output_shapes
:ö*
dtype0ý
BrainNetCNN/Edge-to-Node/Conv2DConv2D6BrainNetCNN/batch_normalization_7/FusedBatchNormV3:y:06BrainNetCNN/Edge-to-Node/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides
¤
/BrainNetCNN/Edge-to-Node/BiasAdd/ReadVariableOpReadVariableOp8brainnetcnn_edge_to_node_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0É
 BrainNetCNN/Edge-to-Node/BiasAddBiasAdd(BrainNetCNN/Edge-to-Node/Conv2D:output:07BrainNetCNN/Edge-to-Node/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
BrainNetCNN/Edge-to-Node/ReluRelu)BrainNetCNN/Edge-to-Node/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö¦
0BrainNetCNN/batch_normalization_8/ReadVariableOpReadVariableOp9brainnetcnn_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0ª
2BrainNetCNN/batch_normalization_8/ReadVariableOp_1ReadVariableOp;brainnetcnn_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0È
ABrainNetCNN/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpJbrainnetcnn_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ì
CBrainNetCNN/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLbrainnetcnn_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
2BrainNetCNN/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3+BrainNetCNN/Edge-to-Node/Relu:activations:08BrainNetCNN/batch_normalization_8/ReadVariableOp:value:0:BrainNetCNN/batch_normalization_8/ReadVariableOp_1:value:0IBrainNetCNN/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0KBrainNetCNN/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿö:::::*
epsilon%o:*
is_training( ±
/BrainNetCNN/Node-to-Graph/Conv2D/ReadVariableOpReadVariableOp8brainnetcnn_node_to_graph_conv2d_readvariableop_resource*'
_output_shapes
:ö *
dtype0þ
 BrainNetCNN/Node-to-Graph/Conv2DConv2D6BrainNetCNN/batch_normalization_8/FusedBatchNormV3:y:07BrainNetCNN/Node-to-Graph/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
¦
0BrainNetCNN/Node-to-Graph/BiasAdd/ReadVariableOpReadVariableOp9brainnetcnn_node_to_graph_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ë
!BrainNetCNN/Node-to-Graph/BiasAddBiasAdd)BrainNetCNN/Node-to-Graph/Conv2D:output:08BrainNetCNN/Node-to-Graph/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
BrainNetCNN/Node-to-Graph/ReluRelu*BrainNetCNN/Node-to-Graph/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
BrainNetCNN/dropout_6/IdentityIdentity,BrainNetCNN/Node-to-Graph/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
BrainNetCNN/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ©
BrainNetCNN/flatten_2/ReshapeReshape'BrainNetCNN/dropout_6/Identity:output:0$BrainNetCNN/flatten_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
%BrainNetCNN/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ì
 BrainNetCNN/concatenate_2/concatConcatV2&BrainNetCNN/flatten_2/Reshape:output:0input_struc.BrainNetCNN/concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
)BrainNetCNN/dense_6/MatMul/ReadVariableOpReadVariableOp2brainnetcnn_dense_6_matmul_readvariableop_resource*
_output_shapes

:#@*
dtype0´
BrainNetCNN/dense_6/MatMulMatMul)BrainNetCNN/concatenate_2/concat:output:01BrainNetCNN/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*BrainNetCNN/dense_6/BiasAdd/ReadVariableOpReadVariableOp3brainnetcnn_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0²
BrainNetCNN/dense_6/BiasAddBiasAdd$BrainNetCNN/dense_6/MatMul:product:02BrainNetCNN/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
BrainNetCNN/dense_6/ReluRelu$BrainNetCNN/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
BrainNetCNN/dropout_7/IdentityIdentity&BrainNetCNN/dense_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)BrainNetCNN/dense_7/MatMul/ReadVariableOpReadVariableOp2brainnetcnn_dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0²
BrainNetCNN/dense_7/MatMulMatMul'BrainNetCNN/dropout_7/Identity:output:01BrainNetCNN/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*BrainNetCNN/dense_7/BiasAdd/ReadVariableOpReadVariableOp3brainnetcnn_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0²
BrainNetCNN/dense_7/BiasAddBiasAdd$BrainNetCNN/dense_7/MatMul:product:02BrainNetCNN/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
BrainNetCNN/dense_7/ReluRelu$BrainNetCNN/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
BrainNetCNN/dropout_8/IdentityIdentity&BrainNetCNN/dense_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)BrainNetCNN/dense_8/MatMul/ReadVariableOpReadVariableOp2brainnetcnn_dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0²
BrainNetCNN/dense_8/MatMulMatMul'BrainNetCNN/dropout_8/Identity:output:01BrainNetCNN/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*BrainNetCNN/dense_8/BiasAdd/ReadVariableOpReadVariableOp3brainnetcnn_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0²
BrainNetCNN/dense_8/BiasAddBiasAdd$BrainNetCNN/dense_8/MatMul:product:02BrainNetCNN/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
BrainNetCNN/dense_8/SigmoidSigmoid$BrainNetCNN/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
IdentityIdentityBrainNetCNN/dense_8/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^BrainNetCNN/Edge-to-Node/BiasAdd/ReadVariableOp/^BrainNetCNN/Edge-to-Node/Conv2D/ReadVariableOp1^BrainNetCNN/Node-to-Graph/BiasAdd/ReadVariableOp0^BrainNetCNN/Node-to-Graph/Conv2D/ReadVariableOpB^BrainNetCNN/batch_normalization_6/FusedBatchNormV3/ReadVariableOpD^BrainNetCNN/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_11^BrainNetCNN/batch_normalization_6/ReadVariableOp3^BrainNetCNN/batch_normalization_6/ReadVariableOp_1B^BrainNetCNN/batch_normalization_7/FusedBatchNormV3/ReadVariableOpD^BrainNetCNN/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_11^BrainNetCNN/batch_normalization_7/ReadVariableOp3^BrainNetCNN/batch_normalization_7/ReadVariableOp_1B^BrainNetCNN/batch_normalization_8/FusedBatchNormV3/ReadVariableOpD^BrainNetCNN/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_11^BrainNetCNN/batch_normalization_8/ReadVariableOp3^BrainNetCNN/batch_normalization_8/ReadVariableOp_1+^BrainNetCNN/dense_6/BiasAdd/ReadVariableOp*^BrainNetCNN/dense_6/MatMul/ReadVariableOp+^BrainNetCNN/dense_7/BiasAdd/ReadVariableOp*^BrainNetCNN/dense_7/MatMul/ReadVariableOp+^BrainNetCNN/dense_8/BiasAdd/ReadVariableOp*^BrainNetCNN/dense_8/MatMul/ReadVariableOp&^BrainNetCNN/e2e_conv_4/ReadVariableOp(^BrainNetCNN/e2e_conv_4/ReadVariableOp_1&^BrainNetCNN/e2e_conv_5/ReadVariableOp(^BrainNetCNN/e2e_conv_5/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿöö:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 2b
/BrainNetCNN/Edge-to-Node/BiasAdd/ReadVariableOp/BrainNetCNN/Edge-to-Node/BiasAdd/ReadVariableOp2`
.BrainNetCNN/Edge-to-Node/Conv2D/ReadVariableOp.BrainNetCNN/Edge-to-Node/Conv2D/ReadVariableOp2d
0BrainNetCNN/Node-to-Graph/BiasAdd/ReadVariableOp0BrainNetCNN/Node-to-Graph/BiasAdd/ReadVariableOp2b
/BrainNetCNN/Node-to-Graph/Conv2D/ReadVariableOp/BrainNetCNN/Node-to-Graph/Conv2D/ReadVariableOp2
ABrainNetCNN/batch_normalization_6/FusedBatchNormV3/ReadVariableOpABrainNetCNN/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2
CBrainNetCNN/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1CBrainNetCNN/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12d
0BrainNetCNN/batch_normalization_6/ReadVariableOp0BrainNetCNN/batch_normalization_6/ReadVariableOp2h
2BrainNetCNN/batch_normalization_6/ReadVariableOp_12BrainNetCNN/batch_normalization_6/ReadVariableOp_12
ABrainNetCNN/batch_normalization_7/FusedBatchNormV3/ReadVariableOpABrainNetCNN/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2
CBrainNetCNN/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1CBrainNetCNN/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12d
0BrainNetCNN/batch_normalization_7/ReadVariableOp0BrainNetCNN/batch_normalization_7/ReadVariableOp2h
2BrainNetCNN/batch_normalization_7/ReadVariableOp_12BrainNetCNN/batch_normalization_7/ReadVariableOp_12
ABrainNetCNN/batch_normalization_8/FusedBatchNormV3/ReadVariableOpABrainNetCNN/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2
CBrainNetCNN/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1CBrainNetCNN/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12d
0BrainNetCNN/batch_normalization_8/ReadVariableOp0BrainNetCNN/batch_normalization_8/ReadVariableOp2h
2BrainNetCNN/batch_normalization_8/ReadVariableOp_12BrainNetCNN/batch_normalization_8/ReadVariableOp_12X
*BrainNetCNN/dense_6/BiasAdd/ReadVariableOp*BrainNetCNN/dense_6/BiasAdd/ReadVariableOp2V
)BrainNetCNN/dense_6/MatMul/ReadVariableOp)BrainNetCNN/dense_6/MatMul/ReadVariableOp2X
*BrainNetCNN/dense_7/BiasAdd/ReadVariableOp*BrainNetCNN/dense_7/BiasAdd/ReadVariableOp2V
)BrainNetCNN/dense_7/MatMul/ReadVariableOp)BrainNetCNN/dense_7/MatMul/ReadVariableOp2X
*BrainNetCNN/dense_8/BiasAdd/ReadVariableOp*BrainNetCNN/dense_8/BiasAdd/ReadVariableOp2V
)BrainNetCNN/dense_8/MatMul/ReadVariableOp)BrainNetCNN/dense_8/MatMul/ReadVariableOp2N
%BrainNetCNN/e2e_conv_4/ReadVariableOp%BrainNetCNN/e2e_conv_4/ReadVariableOp2R
'BrainNetCNN/e2e_conv_4/ReadVariableOp_1'BrainNetCNN/e2e_conv_4/ReadVariableOp_12N
%BrainNetCNN/e2e_conv_5/ReadVariableOp%BrainNetCNN/e2e_conv_5/ReadVariableOp2R
'BrainNetCNN/e2e_conv_5/ReadVariableOp_1'BrainNetCNN/e2e_conv_5/ReadVariableOp_1:\ X
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
#
_user_specified_name	input_img:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_struc
ø
¹
G__inference_Edge-to-Node_layer_call_and_return_conditional_losses_33875

inputs9
conv2d_readvariableop_resource:ö-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:ö*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿöY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:ö*
dtype0¡
&Edge-to-Node/kernel/Regularizer/SquareSquare=Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö~
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
 *
×#<©
#Edge-to-Node/kernel/Regularizer/mulMul.Edge-to-Node/kernel/Regularizer/mul/x:output:0,Edge-to-Node/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö¯
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp6^Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿöö: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2n
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
 
_user_specified_nameinputs
Á

'__inference_dense_6_layer_call_fn_34029

inputs
unknown:#@
	unknown_0:@
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_32372o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
{
ò
E__inference_e2e_conv_5_layer_call_and_return_conditional_losses_32269

inputs2
readvariableop_resource:ö
identity¢ReadVariableOp¢ReadVariableOp_1¢3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOpo
ReadVariableOpReadVariableOpreadvariableop_resource*'
_output_shapes
:ö*
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
valueB"      
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   ö         t
ReshapeReshapestrided_slice:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:öq
ReadVariableOp_1ReadVariableOpreadvariableop_resource*'
_output_shapes
:ö*
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
valueB"      
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_maskh
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ö            z
	Reshape_1Reshapestrided_slice_1:output:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:ö
convolutionConv2DinputsReshape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides

convolution_1Conv2DinputsReshape_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :þ.
concatConcatV2convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0concat/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿööO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :+
concat_1ConcatV2convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0concat_1/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿööl
addAddV2concat:output:0concat_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*'
_output_shapes
:ö*
dtype0
$e2e_conv_5/kernel/Regularizer/SquareSquare;e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö|
#e2e_conv_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¡
!e2e_conv_5/kernel/Regularizer/SumSum(e2e_conv_5/kernel/Regularizer/Square:y:0,e2e_conv_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#e2e_conv_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<£
!e2e_conv_5/kernel/Regularizer/mulMul,e2e_conv_5/kernel/Regularizer/mul/x:output:0*e2e_conv_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö 
NoOpNoOp^ReadVariableOp^ReadVariableOp_14^e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿöö: 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12j
3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
 
_user_specified_nameinputs
õp

F__inference_BrainNetCNN_layer_call_and_return_conditional_losses_32799

inputs
inputs_1+
e2e_conv_4_32711:ö)
batch_normalization_6_32714:)
batch_normalization_6_32716:)
batch_normalization_6_32718:)
batch_normalization_6_32720:+
e2e_conv_5_32723:ö)
batch_normalization_7_32726:)
batch_normalization_7_32728:)
batch_normalization_7_32730:)
batch_normalization_7_32732:-
edge_to_node_32735:ö 
edge_to_node_32737:)
batch_normalization_8_32740:)
batch_normalization_8_32742:)
batch_normalization_8_32744:)
batch_normalization_8_32746:.
node_to_graph_32749:ö !
node_to_graph_32751: 
dense_6_32757:#@
dense_6_32759:@
dense_7_32763:@@
dense_7_32765:@
dense_8_32769:@
dense_8_32771:
identity¢$Edge-to-Node/StatefulPartitionedCall¢5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp¢%Node-to-Graph/StatefulPartitionedCall¢6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢!dropout_6/StatefulPartitionedCall¢!dropout_7/StatefulPartitionedCall¢!dropout_8/StatefulPartitionedCall¢"e2e_conv_4/StatefulPartitionedCall¢3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp¢"e2e_conv_5/StatefulPartitionedCall¢3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOpî
"e2e_conv_4/StatefulPartitionedCallStatefulPartitionedCallinputse2e_conv_4_32711*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_e2e_conv_4_layer_call_and_return_conditional_losses_32222
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall+e2e_conv_4/StatefulPartitionedCall:output:0batch_normalization_6_32714batch_normalization_6_32716batch_normalization_6_32718batch_normalization_6_32720*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_32040
"e2e_conv_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0e2e_conv_5_32723*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_e2e_conv_5_layer_call_and_return_conditional_losses_32269
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall+e2e_conv_5/StatefulPartitionedCall:output:0batch_normalization_7_32726batch_normalization_7_32728batch_normalization_7_32730batch_normalization_7_32732*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_32104¹
$Edge-to-Node/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0edge_to_node_32735edge_to_node_32737*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_Edge-to-Node_layer_call_and_return_conditional_losses_32299
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall-Edge-to-Node/StatefulPartitionedCall:output:0batch_normalization_8_32740batch_normalization_8_32742batch_normalization_8_32744batch_normalization_8_32746*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_32168¼
%Node-to-Graph/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0node_to_graph_32749node_to_graph_32751*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Node-to-Graph_layer_call_and_return_conditional_losses_32331ü
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall.Node-to-Graph/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_32611à
flatten_2/PartitionedCallPartitionedCall*dropout_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_32350ë
concatenate_2/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_32359
dense_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_6_32757dense_6_32759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_32372
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_32565
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_7_32763dense_7_32765*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_32396
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_32532
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_8_32769dense_8_32771*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_32420
3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpe2e_conv_4_32711*'
_output_shapes
:ö*
dtype0
$e2e_conv_4/kernel/Regularizer/SquareSquare;e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö|
#e2e_conv_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¡
!e2e_conv_4/kernel/Regularizer/SumSum(e2e_conv_4/kernel/Regularizer/Square:y:0,e2e_conv_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#e2e_conv_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<£
!e2e_conv_4/kernel/Regularizer/mulMul,e2e_conv_4/kernel/Regularizer/mul/x:output:0*e2e_conv_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpe2e_conv_5_32723*'
_output_shapes
:ö*
dtype0
$e2e_conv_5/kernel/Regularizer/SquareSquare;e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö|
#e2e_conv_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¡
!e2e_conv_5/kernel/Regularizer/SumSum(e2e_conv_5/kernel/Regularizer/Square:y:0,e2e_conv_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#e2e_conv_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<£
!e2e_conv_5/kernel/Regularizer/mulMul,e2e_conv_5/kernel/Regularizer/mul/x:output:0*e2e_conv_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOpReadVariableOpedge_to_node_32735*'
_output_shapes
:ö*
dtype0¡
&Edge-to-Node/kernel/Regularizer/SquareSquare=Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö~
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
 *
×#<©
#Edge-to-Node/kernel/Regularizer/mulMul.Edge-to-Node/kernel/Regularizer/mul/x:output:0,Edge-to-Node/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOpReadVariableOpnode_to_graph_32749*'
_output_shapes
:ö *
dtype0£
'Node-to-Graph/kernel/Regularizer/SquareSquare>Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö 
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
 *
×#<¬
$Node-to-Graph/kernel/Regularizer/mulMul/Node-to-Graph/kernel/Regularizer/mul/x:output:0-Node-to-Graph/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp%^Edge-to-Node/StatefulPartitionedCall6^Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp&^Node-to-Graph/StatefulPartitionedCall7^Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall#^e2e_conv_4/StatefulPartitionedCall4^e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp#^e2e_conv_5/StatefulPartitionedCall4^e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿöö:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 2L
$Edge-to-Node/StatefulPartitionedCall$Edge-to-Node/StatefulPartitionedCall2n
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp2N
%Node-to-Graph/StatefulPartitionedCall%Node-to-Graph/StatefulPartitionedCall2p
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2H
"e2e_conv_4/StatefulPartitionedCall"e2e_conv_4/StatefulPartitionedCall2j
3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp2H
"e2e_conv_5/StatefulPartitionedCall"e2e_conv_5/StatefulPartitionedCall2j
3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë

P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_33716

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á

'__inference_dense_7_layer_call_fn_34076

inputs
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_32396o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ò
b
)__inference_dropout_8_layer_call_fn_34097

inputs
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_32532o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
²

c
D__inference_dropout_6_layer_call_and_return_conditional_losses_32611

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
»l

F__inference_BrainNetCNN_layer_call_and_return_conditional_losses_32451

inputs
inputs_1+
e2e_conv_4_32223:ö)
batch_normalization_6_32226:)
batch_normalization_6_32228:)
batch_normalization_6_32230:)
batch_normalization_6_32232:+
e2e_conv_5_32270:ö)
batch_normalization_7_32273:)
batch_normalization_7_32275:)
batch_normalization_7_32277:)
batch_normalization_7_32279:-
edge_to_node_32300:ö 
edge_to_node_32302:)
batch_normalization_8_32305:)
batch_normalization_8_32307:)
batch_normalization_8_32309:)
batch_normalization_8_32311:.
node_to_graph_32332:ö !
node_to_graph_32334: 
dense_6_32373:#@
dense_6_32375:@
dense_7_32397:@@
dense_7_32399:@
dense_8_32421:@
dense_8_32423:
identity¢$Edge-to-Node/StatefulPartitionedCall¢5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp¢%Node-to-Graph/StatefulPartitionedCall¢6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢"e2e_conv_4/StatefulPartitionedCall¢3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp¢"e2e_conv_5/StatefulPartitionedCall¢3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOpî
"e2e_conv_4/StatefulPartitionedCallStatefulPartitionedCallinputse2e_conv_4_32223*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_e2e_conv_4_layer_call_and_return_conditional_losses_32222
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall+e2e_conv_4/StatefulPartitionedCall:output:0batch_normalization_6_32226batch_normalization_6_32228batch_normalization_6_32230batch_normalization_6_32232*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_32009
"e2e_conv_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0e2e_conv_5_32270*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_e2e_conv_5_layer_call_and_return_conditional_losses_32269
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall+e2e_conv_5/StatefulPartitionedCall:output:0batch_normalization_7_32273batch_normalization_7_32275batch_normalization_7_32277batch_normalization_7_32279*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_32073¹
$Edge-to-Node/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0edge_to_node_32300edge_to_node_32302*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_Edge-to-Node_layer_call_and_return_conditional_losses_32299
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall-Edge-to-Node/StatefulPartitionedCall:output:0batch_normalization_8_32305batch_normalization_8_32307batch_normalization_8_32309batch_normalization_8_32311*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_32137¼
%Node-to-Graph/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0node_to_graph_32332node_to_graph_32334*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Node-to-Graph_layer_call_and_return_conditional_losses_32331ì
dropout_6/PartitionedCallPartitionedCall.Node-to-Graph/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_32342Ø
flatten_2/PartitionedCallPartitionedCall"dropout_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_32350ë
concatenate_2/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_32359
dense_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_6_32373dense_6_32375*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_32372Þ
dropout_7/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_32383
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_7_32397dense_7_32399*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_32396Þ
dropout_8/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_32407
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_8_32421dense_8_32423*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_32420
3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpe2e_conv_4_32223*'
_output_shapes
:ö*
dtype0
$e2e_conv_4/kernel/Regularizer/SquareSquare;e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö|
#e2e_conv_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¡
!e2e_conv_4/kernel/Regularizer/SumSum(e2e_conv_4/kernel/Regularizer/Square:y:0,e2e_conv_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#e2e_conv_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<£
!e2e_conv_4/kernel/Regularizer/mulMul,e2e_conv_4/kernel/Regularizer/mul/x:output:0*e2e_conv_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpe2e_conv_5_32270*'
_output_shapes
:ö*
dtype0
$e2e_conv_5/kernel/Regularizer/SquareSquare;e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö|
#e2e_conv_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¡
!e2e_conv_5/kernel/Regularizer/SumSum(e2e_conv_5/kernel/Regularizer/Square:y:0,e2e_conv_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#e2e_conv_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<£
!e2e_conv_5/kernel/Regularizer/mulMul,e2e_conv_5/kernel/Regularizer/mul/x:output:0*e2e_conv_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOpReadVariableOpedge_to_node_32300*'
_output_shapes
:ö*
dtype0¡
&Edge-to-Node/kernel/Regularizer/SquareSquare=Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö~
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
 *
×#<©
#Edge-to-Node/kernel/Regularizer/mulMul.Edge-to-Node/kernel/Regularizer/mul/x:output:0,Edge-to-Node/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOpReadVariableOpnode_to_graph_32332*'
_output_shapes
:ö *
dtype0£
'Node-to-Graph/kernel/Regularizer/SquareSquare>Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö 
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
 *
×#<¬
$Node-to-Graph/kernel/Regularizer/mulMul/Node-to-Graph/kernel/Regularizer/mul/x:output:0-Node-to-Graph/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp%^Edge-to-Node/StatefulPartitionedCall6^Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp&^Node-to-Graph/StatefulPartitionedCall7^Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall#^e2e_conv_4/StatefulPartitionedCall4^e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp#^e2e_conv_5/StatefulPartitionedCall4^e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿöö:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 2L
$Edge-to-Node/StatefulPartitionedCall$Edge-to-Node/StatefulPartitionedCall2n
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp2N
%Node-to-Graph/StatefulPartitionedCall%Node-to-Graph/StatefulPartitionedCall2p
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2H
"e2e_conv_4/StatefulPartitionedCall"e2e_conv_4/StatefulPartitionedCall2j
3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp2H
"e2e_conv_5/StatefulPartitionedCall"e2e_conv_5/StatefulPartitionedCall2j
3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò	
c
D__inference_dropout_7_layer_call_and_return_conditional_losses_34067

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
³
Ã
__inference_loss_fn_3_34178Z
?node_to_graph_kernel_regularizer_square_readvariableop_resource:ö 
identity¢6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp¿
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOpReadVariableOp?node_to_graph_kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:ö *
dtype0£
'Node-to-Graph/kernel/Regularizer/SquareSquare>Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö 
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
 *
×#<¬
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
Ë

P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_32073

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
½
__inference_loss_fn_1_34156W
<e2e_conv_5_kernel_regularizer_square_readvariableop_resource:ö
identity¢3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp¹
3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<e2e_conv_5_kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:ö*
dtype0
$e2e_conv_5/kernel/Regularizer/SquareSquare;e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö|
#e2e_conv_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¡
!e2e_conv_5/kernel/Regularizer/SumSum(e2e_conv_5/kernel/Regularizer/Square:y:0,e2e_conv_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#e2e_conv_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<£
!e2e_conv_5/kernel/Regularizer/mulMul,e2e_conv_5/kernel/Regularizer/mul/x:output:0*e2e_conv_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
IdentityIdentity%e2e_conv_5/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp
	
Ð
5__inference_batch_normalization_6_layer_call_fn_33698

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_32040
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢

+__inference_BrainNetCNN_layer_call_fn_33226
inputs_0
inputs_1"
unknown:ö
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:$
	unknown_4:ö
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:$
	unknown_9:ö

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:%

unknown_15:ö 

unknown_16: 

unknown_17:#@

unknown_18:@

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_BrainNetCNN_layer_call_and_return_conditional_losses_32799o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿöö:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ë

P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_32009

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò	
c
D__inference_dropout_7_layer_call_and_return_conditional_losses_32565

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ð

*__inference_e2e_conv_5_layer_call_fn_33747

inputs"
unknown:ö
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_e2e_conv_5_layer_call_and_return_conditional_losses_32269y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿöö: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
 
_user_specified_nameinputs
Åó

F__inference_BrainNetCNN_layer_call_and_return_conditional_losses_33569
inputs_0
inputs_1=
"e2e_conv_4_readvariableop_resource:ö;
-batch_normalization_6_readvariableop_resource:=
/batch_normalization_6_readvariableop_1_resource:L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:=
"e2e_conv_5_readvariableop_resource:ö;
-batch_normalization_7_readvariableop_resource:=
/batch_normalization_7_readvariableop_1_resource:L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:F
+edge_to_node_conv2d_readvariableop_resource:ö:
,edge_to_node_biasadd_readvariableop_resource:;
-batch_normalization_8_readvariableop_resource:=
/batch_normalization_8_readvariableop_1_resource:L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:G
,node_to_graph_conv2d_readvariableop_resource:ö ;
-node_to_graph_biasadd_readvariableop_resource: 8
&dense_6_matmul_readvariableop_resource:#@5
'dense_6_biasadd_readvariableop_resource:@8
&dense_7_matmul_readvariableop_resource:@@5
'dense_7_biasadd_readvariableop_resource:@8
&dense_8_matmul_readvariableop_resource:@5
'dense_8_biasadd_readvariableop_resource:
identity¢#Edge-to-Node/BiasAdd/ReadVariableOp¢"Edge-to-Node/Conv2D/ReadVariableOp¢5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp¢$Node-to-Graph/BiasAdd/ReadVariableOp¢#Node-to-Graph/Conv2D/ReadVariableOp¢6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp¢$batch_normalization_6/AssignNewValue¢&batch_normalization_6/AssignNewValue_1¢5batch_normalization_6/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_6/ReadVariableOp¢&batch_normalization_6/ReadVariableOp_1¢$batch_normalization_7/AssignNewValue¢&batch_normalization_7/AssignNewValue_1¢5batch_normalization_7/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_7/ReadVariableOp¢&batch_normalization_7/ReadVariableOp_1¢$batch_normalization_8/AssignNewValue¢&batch_normalization_8/AssignNewValue_1¢5batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_8/ReadVariableOp¢&batch_normalization_8/ReadVariableOp_1¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢e2e_conv_4/ReadVariableOp¢e2e_conv_4/ReadVariableOp_1¢3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp¢e2e_conv_5/ReadVariableOp¢e2e_conv_5/ReadVariableOp_1¢3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp
e2e_conv_4/ReadVariableOpReadVariableOp"e2e_conv_4_readvariableop_resource*'
_output_shapes
:ö*
dtype0o
e2e_conv_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        q
 e2e_conv_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       q
 e2e_conv_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¿
e2e_conv_4/strided_sliceStridedSlice!e2e_conv_4/ReadVariableOp:value:0'e2e_conv_4/strided_slice/stack:output:0)e2e_conv_4/strided_slice/stack_1:output:0)e2e_conv_4/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_maskq
e2e_conv_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   ö         
e2e_conv_4/ReshapeReshape!e2e_conv_4/strided_slice:output:0!e2e_conv_4/Reshape/shape:output:0*
T0*'
_output_shapes
:ö
e2e_conv_4/ReadVariableOp_1ReadVariableOp"e2e_conv_4_readvariableop_resource*'
_output_shapes
:ö*
dtype0q
 e2e_conv_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"e2e_conv_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"e2e_conv_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      É
e2e_conv_4/strided_slice_1StridedSlice#e2e_conv_4/ReadVariableOp_1:value:0)e2e_conv_4/strided_slice_1/stack:output:0+e2e_conv_4/strided_slice_1/stack_1:output:0+e2e_conv_4/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_masks
e2e_conv_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ö            
e2e_conv_4/Reshape_1Reshape#e2e_conv_4/strided_slice_1:output:0#e2e_conv_4/Reshape_1/shape:output:0*
T0*'
_output_shapes
:ö«
e2e_conv_4/convolutionConv2Dinputs_0e2e_conv_4/Reshape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides
¯
e2e_conv_4/convolution_1Conv2Dinputs_0e2e_conv_4/Reshape_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides
X
e2e_conv_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¦D
e2e_conv_4/concatConcatV2!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0!e2e_conv_4/convolution_1:output:0e2e_conv_4/concat/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿööZ
e2e_conv_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :¾@
e2e_conv_4/concat_1ConcatV2e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0e2e_conv_4/convolution:output:0!e2e_conv_4/concat_1/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
e2e_conv_4/addAddV2e2e_conv_4/concat:output:0e2e_conv_4/concat_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype0
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype0°
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0´
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0À
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3e2e_conv_4/add:z:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿöö:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
e2e_conv_5/ReadVariableOpReadVariableOp"e2e_conv_5_readvariableop_resource*'
_output_shapes
:ö*
dtype0o
e2e_conv_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        q
 e2e_conv_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       q
 e2e_conv_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¿
e2e_conv_5/strided_sliceStridedSlice!e2e_conv_5/ReadVariableOp:value:0'e2e_conv_5/strided_slice/stack:output:0)e2e_conv_5/strided_slice/stack_1:output:0)e2e_conv_5/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_maskq
e2e_conv_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   ö         
e2e_conv_5/ReshapeReshape!e2e_conv_5/strided_slice:output:0!e2e_conv_5/Reshape/shape:output:0*
T0*'
_output_shapes
:ö
e2e_conv_5/ReadVariableOp_1ReadVariableOp"e2e_conv_5_readvariableop_resource*'
_output_shapes
:ö*
dtype0q
 e2e_conv_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"e2e_conv_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"e2e_conv_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      É
e2e_conv_5/strided_slice_1StridedSlice#e2e_conv_5/ReadVariableOp_1:value:0)e2e_conv_5/strided_slice_1/stack:output:0+e2e_conv_5/strided_slice_1/stack_1:output:0+e2e_conv_5/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_masks
e2e_conv_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ö            
e2e_conv_5/Reshape_1Reshape#e2e_conv_5/strided_slice_1:output:0#e2e_conv_5/Reshape_1/shape:output:0*
T0*'
_output_shapes
:öÍ
e2e_conv_5/convolutionConv2D*batch_normalization_6/FusedBatchNormV3:y:0e2e_conv_5/Reshape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides
Ñ
e2e_conv_5/convolution_1Conv2D*batch_normalization_6/FusedBatchNormV3:y:0e2e_conv_5/Reshape_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides
X
e2e_conv_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¦D
e2e_conv_5/concatConcatV2!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0!e2e_conv_5/convolution_1:output:0e2e_conv_5/concat/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿööZ
e2e_conv_5/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :¾@
e2e_conv_5/concat_1ConcatV2e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0e2e_conv_5/convolution:output:0!e2e_conv_5/concat_1/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
e2e_conv_5/addAddV2e2e_conv_5/concat:output:0e2e_conv_5/concat_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype0
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype0°
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0´
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0À
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3e2e_conv_5/add:z:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿöö:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
"Edge-to-Node/Conv2D/ReadVariableOpReadVariableOp+edge_to_node_conv2d_readvariableop_resource*'
_output_shapes
:ö*
dtype0Ù
Edge-to-Node/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0*Edge-to-Node/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides

#Edge-to-Node/BiasAdd/ReadVariableOpReadVariableOp,edge_to_node_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¥
Edge-to-Node/BiasAddBiasAddEdge-to-Node/Conv2D:output:0+Edge-to-Node/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿös
Edge-to-Node/ReluReluEdge-to-Node/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0°
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0´
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ì
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3Edge-to-Node/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿö:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
#Node-to-Graph/Conv2D/ReadVariableOpReadVariableOp,node_to_graph_conv2d_readvariableop_resource*'
_output_shapes
:ö *
dtype0Ú
Node-to-Graph/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0+Node-to-Graph/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

$Node-to-Graph/BiasAdd/ReadVariableOpReadVariableOp-node_to_graph_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0§
Node-to-Graph/BiasAddBiasAddNode-to-Graph/Conv2D:output:0,Node-to-Graph/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
Node-to-Graph/ReluReluNode-to-Graph/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout_6/dropout/MulMul Node-to-Graph/Relu:activations:0 dropout_6/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
dropout_6/dropout/ShapeShape Node-to-Graph/Relu:activations:0*
T0*
_output_shapes
:¨
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0e
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ì
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
flatten_2/ReshapeReshapedropout_6/dropout/Mul_1:z:0flatten_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¥
concatenate_2/concatConcatV2flatten_2/Reshape:output:0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:#@*
dtype0
dense_6/MatMulMatMulconcatenate_2/concat:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
dropout_7/dropout/MulMuldense_6/Relu:activations:0 dropout_7/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
dropout_7/dropout/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
: 
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0e
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ä
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0
dense_7/MatMulMatMuldropout_7/dropout/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
dropout_8/dropout/MulMuldense_7/Relu:activations:0 dropout_8/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
dropout_8/dropout/ShapeShapedense_7/Relu:activations:0*
T0*
_output_shapes
: 
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0e
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ä
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_8/MatMulMatMuldropout_8/dropout/Mul_1:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_8/SigmoidSigmoiddense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"e2e_conv_4_readvariableop_resource*'
_output_shapes
:ö*
dtype0
$e2e_conv_4/kernel/Regularizer/SquareSquare;e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö|
#e2e_conv_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¡
!e2e_conv_4/kernel/Regularizer/SumSum(e2e_conv_4/kernel/Regularizer/Square:y:0,e2e_conv_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#e2e_conv_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<£
!e2e_conv_4/kernel/Regularizer/mulMul,e2e_conv_4/kernel/Regularizer/mul/x:output:0*e2e_conv_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"e2e_conv_5_readvariableop_resource*'
_output_shapes
:ö*
dtype0
$e2e_conv_5/kernel/Regularizer/SquareSquare;e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö|
#e2e_conv_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¡
!e2e_conv_5/kernel/Regularizer/SumSum(e2e_conv_5/kernel/Regularizer/Square:y:0,e2e_conv_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#e2e_conv_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<£
!e2e_conv_5/kernel/Regularizer/mulMul,e2e_conv_5/kernel/Regularizer/mul/x:output:0*e2e_conv_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ª
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+edge_to_node_conv2d_readvariableop_resource*'
_output_shapes
:ö*
dtype0¡
&Edge-to-Node/kernel/Regularizer/SquareSquare=Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö~
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
 *
×#<©
#Edge-to-Node/kernel/Regularizer/mulMul.Edge-to-Node/kernel/Regularizer/mul/x:output:0,Edge-to-Node/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¬
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOpReadVariableOp,node_to_graph_conv2d_readvariableop_resource*'
_output_shapes
:ö *
dtype0£
'Node-to-Graph/kernel/Regularizer/SquareSquare>Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö 
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
 *
×#<¬
$Node-to-Graph/kernel/Regularizer/mulMul/Node-to-Graph/kernel/Regularizer/mul/x:output:0-Node-to-Graph/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentitydense_8/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
NoOpNoOp$^Edge-to-Node/BiasAdd/ReadVariableOp#^Edge-to-Node/Conv2D/ReadVariableOp6^Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp%^Node-to-Graph/BiasAdd/ReadVariableOp$^Node-to-Graph/Conv2D/ReadVariableOp7^Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^e2e_conv_4/ReadVariableOp^e2e_conv_4/ReadVariableOp_14^e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp^e2e_conv_5/ReadVariableOp^e2e_conv_5/ReadVariableOp_14^e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿöö:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 2J
#Edge-to-Node/BiasAdd/ReadVariableOp#Edge-to-Node/BiasAdd/ReadVariableOp2H
"Edge-to-Node/Conv2D/ReadVariableOp"Edge-to-Node/Conv2D/ReadVariableOp2n
5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp5Edge-to-Node/kernel/Regularizer/Square/ReadVariableOp2L
$Node-to-Graph/BiasAdd/ReadVariableOp$Node-to-Graph/BiasAdd/ReadVariableOp2J
#Node-to-Graph/Conv2D/ReadVariableOp#Node-to-Graph/Conv2D/ReadVariableOp2p
6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp6Node-to-Graph/kernel/Regularizer/Square/ReadVariableOp2L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp26
e2e_conv_4/ReadVariableOpe2e_conv_4/ReadVariableOp2:
e2e_conv_4/ReadVariableOp_1e2e_conv_4/ReadVariableOp_12j
3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp26
e2e_conv_5/ReadVariableOpe2e_conv_5/ReadVariableOp2:
e2e_conv_5/ReadVariableOp_1e2e_conv_5/ReadVariableOp_12j
3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp3e2e_conv_5/kernel/Regularizer/Square/ReadVariableOp:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ë

P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_32137

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
¿
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_32040

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
¿
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_32104

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_32383

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
Ð
5__inference_batch_normalization_6_layer_call_fn_33685

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_32009
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
{
ò
E__inference_e2e_conv_4_layer_call_and_return_conditional_losses_33672

inputs2
readvariableop_resource:ö
identity¢ReadVariableOp¢ReadVariableOp_1¢3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOpo
ReadVariableOpReadVariableOpreadvariableop_resource*'
_output_shapes
:ö*
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
valueB"      
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   ö         t
ReshapeReshapestrided_slice:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:öq
ReadVariableOp_1ReadVariableOpreadvariableop_resource*'
_output_shapes
:ö*
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
valueB"      
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ö*

begin_mask*
end_mask*
shrink_axis_maskh
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ö            z
	Reshape_1Reshapestrided_slice_1:output:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:ö
convolutionConv2DinputsReshape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides

convolution_1Conv2DinputsReshape_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö*
paddingVALID*
strides
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :þ.
concatConcatV2convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0convolution_1:output:0concat/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿööO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :+
concat_1ConcatV2convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0convolution:output:0concat_1/axis:output:0*
Nö*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿööl
addAddV2concat:output:0concat_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*'
_output_shapes
:ö*
dtype0
$e2e_conv_4/kernel/Regularizer/SquareSquare;e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:ö|
#e2e_conv_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¡
!e2e_conv_4/kernel/Regularizer/SumSum(e2e_conv_4/kernel/Regularizer/Square:y:0,e2e_conv_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#e2e_conv_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<£
!e2e_conv_4/kernel/Regularizer/mulMul,e2e_conv_4/kernel/Regularizer/mul/x:output:0*e2e_conv_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö 
NoOpNoOp^ReadVariableOp^ReadVariableOp_14^e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿöö: 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12j
3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp3e2e_conv_4/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿöö
 
_user_specified_nameinputs
Ù
¿
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_33734

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
E
)__inference_flatten_2_layer_call_fn_34001

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_32350`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
Ð
5__inference_batch_normalization_8_layer_call_fn_33888

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_32137
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ý
serving_defaulté
I
	input_img<
serving_default_input_img:0ÿÿÿÿÿÿÿÿÿöö
C
input_struc4
serving_default_input_struc:0ÿÿÿÿÿÿÿÿÿ;
dense_80
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ì©
Ë
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer-15
layer_with_weights-9
layer-16
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
±

kernel
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
"axis
	#gamma
$beta
%moving_mean
&moving_variance
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
±

-kernel
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
4axis
	5gamma
6beta
7moving_mean
8moving_variance
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer
»

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Rkernel
Sbias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^_random_generator
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
¥
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
»

mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y_random_generator
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
¿

|kernel
}bias
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ù
	iter
beta_1
beta_2

decay
learning_rate
momentum_cachemþ#mÿ$m-m5m6m?m@mHmImRmSmmmnm|m}m	m	mv#v$v-v5v6v?v@vHvIvRvSvmvnv|v}v	v 	v¡"
	optimizer
Ø
0
#1
$2
%3
&4
-5
56
67
78
89
?10
@11
H12
I13
J14
K15
R16
S17
m18
n19
|20
}21
22
23"
trackable_list_wrapper
¨
0
#1
$2
-3
54
65
?6
@7
H8
I9
R10
S11
m12
n13
|14
}15
16
17"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
Ï
non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ú2÷
+__inference_BrainNetCNN_layer_call_fn_32502
+__inference_BrainNetCNN_layer_call_fn_33172
+__inference_BrainNetCNN_layer_call_fn_33226
+__inference_BrainNetCNN_layer_call_fn_32904À
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
æ2ã
F__inference_BrainNetCNN_layer_call_and_return_conditional_losses_33387
F__inference_BrainNetCNN_layer_call_and_return_conditional_losses_33569
F__inference_BrainNetCNN_layer_call_and_return_conditional_losses_32996
F__inference_BrainNetCNN_layer_call_and_return_conditional_losses_33088À
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
ÚB×
 __inference__wrapped_model_31987	input_imginput_struc"
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
-
¢serving_default"
signature_map
,:*ö2e2e_conv_4/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
²
£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_e2e_conv_4_layer_call_fn_33638¢
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
ï2ì
E__inference_e2e_conv_4_layer_call_and_return_conditional_losses_33672¢
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
):'2batch_normalization_6/gamma
(:&2batch_normalization_6/beta
1:/ (2!batch_normalization_6/moving_mean
5:3 (2%batch_normalization_6/moving_variance
<
#0
$1
%2
&3"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
¨2¥
5__inference_batch_normalization_6_layer_call_fn_33685
5__inference_batch_normalization_6_layer_call_fn_33698´
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
Þ2Û
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_33716
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_33734´
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
,:*ö2e2e_conv_5/kernel
'
-0"
trackable_list_wrapper
'
-0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
²
­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_e2e_conv_5_layer_call_fn_33747¢
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
ï2ì
E__inference_e2e_conv_5_layer_call_and_return_conditional_losses_33781¢
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
):'2batch_normalization_7/gamma
(:&2batch_normalization_7/beta
1:/ (2!batch_normalization_7/moving_mean
5:3 (2%batch_normalization_7/moving_variance
<
50
61
72
83"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
²
²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
¨2¥
5__inference_batch_normalization_7_layer_call_fn_33794
5__inference_batch_normalization_7_layer_call_fn_33807´
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
Þ2Û
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_33825
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_33843´
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
.:,ö2Edge-to-Node/kernel
:2Edge-to-Node/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
²
·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_Edge-to-Node_layer_call_fn_33858¢
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
G__inference_Edge-to-Node_layer_call_and_return_conditional_losses_33875¢
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
):'2batch_normalization_8/gamma
(:&2batch_normalization_8/beta
1:/ (2!batch_normalization_8/moving_mean
5:3 (2%batch_normalization_8/moving_variance
<
H0
I1
J2
K3"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
¨2¥
5__inference_batch_normalization_8_layer_call_fn_33888
5__inference_batch_normalization_8_layer_call_fn_33901´
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
Þ2Û
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_33919
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_33937´
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
/:-ö 2Node-to-Graph/kernel
 : 2Node-to-Graph/bias
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
²
Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_Node-to-Graph_layer_call_fn_33952¢
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
H__inference_Node-to-Graph_layer_call_and_return_conditional_losses_33969¢
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
Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_dropout_6_layer_call_fn_33974
)__inference_dropout_6_layer_call_fn_33979´
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
Æ2Ã
D__inference_dropout_6_layer_call_and_return_conditional_losses_33984
D__inference_dropout_6_layer_call_and_return_conditional_losses_33996´
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
Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_flatten_2_layer_call_fn_34001¢
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
D__inference_flatten_2_layer_call_and_return_conditional_losses_34007¢
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
Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_concatenate_2_layer_call_fn_34013¢
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
H__inference_concatenate_2_layer_call_and_return_conditional_losses_34020¢
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
 :#@2dense_6/kernel
:@2dense_6/bias
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense_6_layer_call_fn_34029¢
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
ì2é
B__inference_dense_6_layer_call_and_return_conditional_losses_34040¢
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
Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_dropout_7_layer_call_fn_34045
)__inference_dropout_7_layer_call_fn_34050´
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
Æ2Ã
D__inference_dropout_7_layer_call_and_return_conditional_losses_34055
D__inference_dropout_7_layer_call_and_return_conditional_losses_34067´
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
 :@@2dense_7/kernel
:@2dense_7/bias
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
¶
ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense_7_layer_call_fn_34076¢
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
ì2é
B__inference_dense_7_layer_call_and_return_conditional_losses_34087¢
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
¸
änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_dropout_8_layer_call_fn_34092
)__inference_dropout_8_layer_call_fn_34097´
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
Æ2Ã
D__inference_dropout_8_layer_call_and_return_conditional_losses_34102
D__inference_dropout_8_layer_call_and_return_conditional_losses_34114´
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
 :@2dense_8/kernel
:2dense_8/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
énon_trainable_variables
êlayers
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense_8_layer_call_fn_34123¢
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
ì2é
B__inference_dense_8_layer_call_and_return_conditional_losses_34134¢
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
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
²2¯
__inference_loss_fn_0_34145
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
²2¯
__inference_loss_fn_1_34156
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
²2¯
__inference_loss_fn_2_34167
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
²2¯
__inference_loss_fn_3_34178
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
J
%0
&1
72
83
J4
K5"
trackable_list_wrapper

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
13
14
15
16"
trackable_list_wrapper
8
î0
ï1
ð2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
×BÔ
#__inference_signature_wrapper_33625	input_imginput_struc"
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
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
%0
&1"
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
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
70
81"
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
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
J0
K1"
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
(
0"
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

ñtotal

òcount
ó	variables
ô	keras_api"
_tf_keras_metric
c

õtotal

öcount
÷
_fn_kwargs
ø	variables
ù	keras_api"
_tf_keras_metric
]
ú
thresholds
ûaccumulator
ü	variables
ý	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
ñ0
ò1"
trackable_list_wrapper
.
ó	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
õ0
ö1"
trackable_list_wrapper
.
ø	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
û0"
trackable_list_wrapper
.
ü	variables"
_generic_user_object
2:0ö2Nadam/e2e_conv_4/kernel/m
/:-2#Nadam/batch_normalization_6/gamma/m
.:,2"Nadam/batch_normalization_6/beta/m
2:0ö2Nadam/e2e_conv_5/kernel/m
/:-2#Nadam/batch_normalization_7/gamma/m
.:,2"Nadam/batch_normalization_7/beta/m
4:2ö2Nadam/Edge-to-Node/kernel/m
%:#2Nadam/Edge-to-Node/bias/m
/:-2#Nadam/batch_normalization_8/gamma/m
.:,2"Nadam/batch_normalization_8/beta/m
5:3ö 2Nadam/Node-to-Graph/kernel/m
&:$ 2Nadam/Node-to-Graph/bias/m
&:$#@2Nadam/dense_6/kernel/m
 :@2Nadam/dense_6/bias/m
&:$@@2Nadam/dense_7/kernel/m
 :@2Nadam/dense_7/bias/m
&:$@2Nadam/dense_8/kernel/m
 :2Nadam/dense_8/bias/m
2:0ö2Nadam/e2e_conv_4/kernel/v
/:-2#Nadam/batch_normalization_6/gamma/v
.:,2"Nadam/batch_normalization_6/beta/v
2:0ö2Nadam/e2e_conv_5/kernel/v
/:-2#Nadam/batch_normalization_7/gamma/v
.:,2"Nadam/batch_normalization_7/beta/v
4:2ö2Nadam/Edge-to-Node/kernel/v
%:#2Nadam/Edge-to-Node/bias/v
/:-2#Nadam/batch_normalization_8/gamma/v
.:,2"Nadam/batch_normalization_8/beta/v
5:3ö 2Nadam/Node-to-Graph/kernel/v
&:$ 2Nadam/Node-to-Graph/bias/v
&:$#@2Nadam/dense_6/kernel/v
 :@2Nadam/dense_6/bias/v
&:$@@2Nadam/dense_7/kernel/v
 :@2Nadam/dense_7/bias/v
&:$@2Nadam/dense_8/kernel/v
 :2Nadam/dense_8/bias/v
F__inference_BrainNetCNN_layer_call_and_return_conditional_losses_32996µ#$%&-5678?@HIJKRSmn|}p¢m
f¢c
YV
-*
	input_imgÿÿÿÿÿÿÿÿÿöö
%"
input_strucÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
F__inference_BrainNetCNN_layer_call_and_return_conditional_losses_33088µ#$%&-5678?@HIJKRSmn|}p¢m
f¢c
YV
-*
	input_imgÿÿÿÿÿÿÿÿÿöö
%"
input_strucÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ü
F__inference_BrainNetCNN_layer_call_and_return_conditional_losses_33387±#$%&-5678?@HIJKRSmn|}l¢i
b¢_
UR
,)
inputs/0ÿÿÿÿÿÿÿÿÿöö
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ü
F__inference_BrainNetCNN_layer_call_and_return_conditional_losses_33569±#$%&-5678?@HIJKRSmn|}l¢i
b¢_
UR
,)
inputs/0ÿÿÿÿÿÿÿÿÿöö
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ø
+__inference_BrainNetCNN_layer_call_fn_32502¨#$%&-5678?@HIJKRSmn|}p¢m
f¢c
YV
-*
	input_imgÿÿÿÿÿÿÿÿÿöö
%"
input_strucÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿØ
+__inference_BrainNetCNN_layer_call_fn_32904¨#$%&-5678?@HIJKRSmn|}p¢m
f¢c
YV
-*
	input_imgÿÿÿÿÿÿÿÿÿöö
%"
input_strucÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÔ
+__inference_BrainNetCNN_layer_call_fn_33172¤#$%&-5678?@HIJKRSmn|}l¢i
b¢_
UR
,)
inputs/0ÿÿÿÿÿÿÿÿÿöö
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÔ
+__inference_BrainNetCNN_layer_call_fn_33226¤#$%&-5678?@HIJKRSmn|}l¢i
b¢_
UR
,)
inputs/0ÿÿÿÿÿÿÿÿÿöö
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿº
G__inference_Edge-to-Node_layer_call_and_return_conditional_losses_33875o?@9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿöö
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿö
 
,__inference_Edge-to-Node_layer_call_fn_33858b?@9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿöö
ª "!ÿÿÿÿÿÿÿÿÿö¹
H__inference_Node-to-Graph_layer_call_and_return_conditional_losses_33969mRS8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿö
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
-__inference_Node-to-Graph_layer_call_fn_33952`RS8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿö
ª " ÿÿÿÿÿÿÿÿÿ Þ
 __inference__wrapped_model_31987¹#$%&-5678?@HIJKRSmn|}h¢e
^¢[
YV
-*
	input_imgÿÿÿÿÿÿÿÿÿöö
%"
input_strucÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_8!
dense_8ÿÿÿÿÿÿÿÿÿë
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_33716#$%&M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ë
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_33734#$%&M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
5__inference_batch_normalization_6_layer_call_fn_33685#$%&M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
5__inference_batch_normalization_6_layer_call_fn_33698#$%&M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿë
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_338255678M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ë
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_338435678M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
5__inference_batch_normalization_7_layer_call_fn_337945678M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
5__inference_batch_normalization_7_layer_call_fn_338075678M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿë
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_33919HIJKM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ë
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_33937HIJKM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
5__inference_batch_normalization_8_layer_call_fn_33888HIJKM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
5__inference_batch_normalization_8_layer_call_fn_33901HIJKM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÐ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_34020Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ 
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ#
 §
-__inference_concatenate_2_layer_call_fn_34013vZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ 
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ#¢
B__inference_dense_6_layer_call_and_return_conditional_losses_34040\mn/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ#
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 z
'__inference_dense_6_layer_call_fn_34029Omn/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ#
ª "ÿÿÿÿÿÿÿÿÿ@¢
B__inference_dense_7_layer_call_and_return_conditional_losses_34087\|}/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 z
'__inference_dense_7_layer_call_fn_34076O|}/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@¤
B__inference_dense_8_layer_call_and_return_conditional_losses_34134^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
'__inference_dense_8_layer_call_fn_34123Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ´
D__inference_dropout_6_layer_call_and_return_conditional_losses_33984l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 ´
D__inference_dropout_6_layer_call_and_return_conditional_losses_33996l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_dropout_6_layer_call_fn_33974_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª " ÿÿÿÿÿÿÿÿÿ 
)__inference_dropout_6_layer_call_fn_33979_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª " ÿÿÿÿÿÿÿÿÿ ¤
D__inference_dropout_7_layer_call_and_return_conditional_losses_34055\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¤
D__inference_dropout_7_layer_call_and_return_conditional_losses_34067\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
)__inference_dropout_7_layer_call_fn_34045O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@|
)__inference_dropout_7_layer_call_fn_34050O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dropout_8_layer_call_and_return_conditional_losses_34102\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¤
D__inference_dropout_8_layer_call_and_return_conditional_losses_34114\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
)__inference_dropout_8_layer_call_fn_34092O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@|
)__inference_dropout_8_layer_call_fn_34097O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@¸
E__inference_e2e_conv_4_layer_call_and_return_conditional_losses_33672o9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿöö
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿöö
 
*__inference_e2e_conv_4_layer_call_fn_33638b9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿöö
ª ""ÿÿÿÿÿÿÿÿÿöö¸
E__inference_e2e_conv_5_layer_call_and_return_conditional_losses_33781o-9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿöö
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿöö
 
*__inference_e2e_conv_5_layer_call_fn_33747b-9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿöö
ª ""ÿÿÿÿÿÿÿÿÿöö¨
D__inference_flatten_2_layer_call_and_return_conditional_losses_34007`7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_flatten_2_layer_call_fn_34001S7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ :
__inference_loss_fn_0_34145¢

¢ 
ª " :
__inference_loss_fn_1_34156-¢

¢ 
ª " :
__inference_loss_fn_2_34167?¢

¢ 
ª " :
__inference_loss_fn_3_34178R¢

¢ 
ª " ø
#__inference_signature_wrapper_33625Ð#$%&-5678?@HIJKRSmn|}¢|
¢ 
uªr
:
	input_img-*
	input_imgÿÿÿÿÿÿÿÿÿöö
4
input_struc%"
input_strucÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_8!
dense_8ÿÿÿÿÿÿÿÿÿ