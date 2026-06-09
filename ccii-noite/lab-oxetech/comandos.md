# comandos.md

## AWS

```bash
### aws criar bucket
aws s3api create-bucket --bucket cloud-lab-$(aws sts get-caller-identity --query Account --output text)-$(date +%s) --region us-east-1

## save o nome do backet para usar em outras operações ou pegar o nome do bucket dentro do localstack
export BUCKET_NAME=cloud-lab-$(aws sts get-caller-identity --query Account --output text)-$(date +%s)

### Copiar arquivo txt
echo "Hello Cloud" > arquivo.txt && aws s3 cp arquivo.txt s3://${BUCKET_NAME}/dados/arquivo.txt

### Copiar arquivo de imagem
aws s3api put-object \ --bucket ${BUCKET_NAME} \ --key image.jpg \ --body image.jpg

### Versionamentos
aws s3api put-bucket-versioning --bucket ${BUCKET_NAME} --versioning-configuration Status=Enabled

### Link temporário
aws s3 presign s3://${BUCKET_NAME}/dados/arquivo.txt --expires-in 3600

### Block publico
aws s3api put-public-access-block --bucket ${BUCKET_NAME} --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

### Verificar o Bloqueio
aws s3api get-public-access-block --bucket ${BUCKET_NAME}

```

## LOCASTACK

```bash
export GATEWAY_LISTEN="0.0.0.0:4566"
export LOCALSTACK_HOST="localhost.localstack.cloud:4566"
```

```bash
## subir o localstack
localstack start -d

## subir o s3 com terraform
tflocal plan

## subir o s3 com terraform
tflocal apply

## save o nome do backet para usar em outras operações ou pegar o nome do bucket dentro do localstack
export BUCKET_NAME="s3-oxetech-local-deployment-bucket"
∂
### Copiar arquivo txt
aws s3 cp arquivo.txt s3://${BUCKET_NAME}/dados/arqsuivo.txt

### Versionamentos
aws s3api put-bucket-versioning --bucket ${BUCKET_NAME} --versioning-configuration Status=Enabled

### Copiar arquivo de Imagem
aws s3 cp arquivo.png s3://${BUCKET_NAME}/image/arquivo.png

### Link temporário
aws s3 presign s3://${BUCKET_NAME}/dados/arquivo.txt --expires-in 3600

### Block publico
aws s3api put-public-access-block --bucket ${BUCKET_NAME} --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

### Verificar o Bloqueio
aws s3api get-public-access-block --bucket ${BUCKET_NAME}

```
