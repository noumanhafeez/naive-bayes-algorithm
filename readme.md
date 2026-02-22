# Mushroom Dataset

# Detail of features of dataset:

## Attribute Information: (classes: edible=e, poisonous=p)

## cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s

## cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s

## cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y

## bruises: bruises=t,no=f

## odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s

## gill-attachment: attached=a,descending=d,free=f,notched=n

## gill-spacing: close=c,crowded=w,distant=d

## gill-size: broad=b,narrow=n

## gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y

## stalk-shape: enlarging=e,tapering=t

## stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?

## stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s

## stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s

## stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

## stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

## veil-type: partial=p,universal=u

## veil-color: brown=n,orange=o,white=w,yellow=y

## ring-number: none=n,one=o,two=t

## ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z

## spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y

## population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y

## habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

## There are 23 columns which are features to identify whether the mushroom is edible or poissnous.
## And all features are categorical and unique. Which means, We don't need to do hard cleaning or pre-processing the
## data. As features are in object datatype. So, we will encode them using labelencoding and then convert them to
## int type.

## Now, We will apply labelencoding to features one by one. 
## NOTE: Label Encoding do label the data in alphabetical order. Means, a:0, b:1, c:3 ... etc. 
## Indeed, you can see this from code and see manually, how label encoding are labeling the features.