%%CMU method
pr=0.49491739;
sigma1=5*pr;
sigma2=12;
Kn=10;
efsi=0.7;
str1=strcat ('E:\compile document\matlab\data\UWA3M\chicken\', name1,'.ply');
str2=strcat ('E:\compile document\matlab\data\UWA3M\chicken\', name2,'.ply');
pcloud1=pcread(str1);
pcloud2=pcread(str2);
PC1=pcloud1.Location;
PC2=pcloud2.Location;
% plot3(PC1(:,1),PC1(:,2),PC1(:,3),'.b','MarkerSize',1);
% hold on;
% plot3(PC2(:,1),PC2(:,2),PC2(:,3),'.r','MarkerSize',1);
% set(gca,'DataAspectRatio',[1 1 1]);
% axis off
keypointcloud1 = pcdownsample(pcloud1,'gridAverage',7*pr);
keypoint1=keypointcloud1.Location;
[n1 m1]=size(keypoint1);
RR=15*pr;
[idx1,dist1]=rangesearch(PC1,keypoint1,RR);
MV1=[];
MDV1=[];
for i=1:n1
    KNN=PC1(idx1{i},:);
    d=dist1{i};
    [V] = LRF_TriLCI(KNN,RR,d,keypoint1(i,:));
    MV1=[MV1;V];
    [DV] = Tri_LCI(KNN,keypoint1(i,:),V,RR);
    MDV1=[MDV1;DV];
end
keypointcloud2 = pcdownsample(pcloud2,'gridAverage',7*pr);
keypoint2=keypointcloud2.Location;
[n2 m2]=size(keypoint2);
[idx2,dist2]=rangesearch(PC2,keypoint2,RR);
MV2=[];
MDV2=[];
for i=1:n2
    KNN=PC2(idx2{i},:);
    d=dist2{i};
    [V] = LRF_TriLCI(KNN,RR,d,keypoint2(i,:));
    MV2=[MV2;V];
    [DV] = Tri_LCI(KNN,keypoint2(i,:),V,RR);
    MDV2=[MDV2;DV];
end
[idxx distt]=knnsearch(MDV1,MDV2,'k',2);
Mmatch=[];
for i=1:n2
    if distt(i,1)/distt(i,2)<=0.9
        match=[idxx(i,1) i];
        Mmatch=[Mmatch;match];
    end
end
[n3 m3]=size(Mmatch);
%%Mmatch denotes the initial correspondence set.
%%start our CMU method
CM=eye(n3);
for i=1:n3-1
    ps1=keypoint1(Mmatch(i,1),:);
    pt1=keypoint2(Mmatch(i,2),:);
    Vs1=MV1(3*Mmatch(i,1)-2:3*Mmatch(i,1),3);
    Vt1=MV2(3*Mmatch(i,2)-2:3*Mmatch(i,2),3);
    for j=i+1:n3
        ps2=keypoint1(Mmatch(j,1),:);
        pt2=keypoint2(Mmatch(j,2),:);
        Vs2=MV1(3*Mmatch(j,1)-2:3*Mmatch(j,1),3);
        Vt2=MV2(3*Mmatch(j,2)-2:3*Mmatch(j,2),3);
        ds=norm(ps1-ps2);
        dt=norm(pt1-pt2);
        dd=abs(ds-dt);
        CC1=exp(-dd*dd/(2*sigma1*sigma1));      %%参数
        if norm(ps2-ps1)==0
            CC2=0;
        else
            As1=real(acos((ps2-ps1)*Vs1/norm(ps2-ps1))*180/pi);
            As2=real(acos((ps1-ps2)*Vs2/norm(ps1-ps2))*180/pi);
            As3=real(acos(Vs1'*Vs2)*180/pi);
            At1=real(acos((pt2-pt1)*Vt1/norm(pt2-pt1))*180/pi);
            At2=real(acos((pt1-pt2)*Vt2/norm(pt1-pt2))*180/pi);
            At3=real(acos(Vt1'*Vt2)*180/pi);
            if As1>90
                As1=180-As1;
            end
            if As2>90
                As2=180-As2;
            end
            if As3>90
                As3=180-As3;
            end
            if At1>90
                At1=180-At1;
            end
            if At2>90
                At2=180-At2;
            end
            if At3>90
                At3=180-At3;
            end
            dA=norm([As1 As2 As3]-[At1 At2 At3]);
            CC2=exp(-dA*dA/(2*sigma2*sigma2));      %%参数
        end
        CC=min(CC1,CC2);
        % CC=CC1;   %%仅GC
        CM(i,j)=CC;
        CM(j,i)=CC;
    end
end
CMrow=sum(CM,2);
%%兼容性矩阵更�?
[F,idx]=sort(CMrow,'descend');
K=n3;
for i=1:10000
    K=floor(0.8*K);
    CK=idx(1:K);
    CMU=CM(:,CK);
    CMUrow=sum(CMU,2);
    [F,idx]=sort(CMUrow,'descend');
    if K<Kn     %%参数
        break
    end
end
id=find(CMUrow>efsi*K);    %%参数
%%无兼容新矩阵更新
% id=find(CMrow>efsi*10);
Cinlier=Mmatch(id,:);
%%Cinlier denotes the selected correspondences.
%%Start to estimate the 3D transformaiton by robust least squares
[n4 m4]=size(Cinlier);
if n4>3
    A=keypoint1(Cinlier(:,1),:);
    Y=keypoint2(Cinlier(:,2),:);
    W0=ones(1,n4);
    R0=zeros(3,3);
    t0=zeros(1,3);
    for i=1:1000
        uA=[W0*A(:,1)/sum(W0) W0*A(:,2)/sum(W0) W0*A(:,3)/sum(W0)];
        uY=[W0*Y(:,1)/sum(W0) W0*Y(:,2)/sum(W0) W0*Y(:,3)/sum(W0)];
        H=zeros(3);
        for j=1:n4
            H=H+(A(j,:)-uA)'*W0(j)*(Y(j,:)-uY);
        end
        [U S V]=svd(H);
        D=diag([1 1 det(U*V')]);
        R=V*D*U';
        t=uY-uA*R';
        Res=Y-A*R'-ones(n4,1)*t;
        clear dd;
        for j=1:n4
            dd(j)=norm(Res(j,:));
        end
        MAD=1.483*median(abs(dd-median(dd)*ones(1,n4)));
        Lb=(dd-median(dd)*ones(1,n4))/MAD;
        clear W;
        for j=1:n4
            k0=1.5;               %参数
            k1=2;
            if abs(Lb(j))<=k0;
                W(j)=1;
            elseif abs(Lb(j))>k0 && abs(Lb(j))<k1;
                W(j)=(k0/abs(Lb(j)))*((k1-abs(Lb(j)))/(k1-k0))*((k1-abs(Lb(j)))/(k1-k0));
            elseif abs(Lb(j))>=k1;
                W(j)=0;
            end
        end
        if norm(R-R0)+norm(t-t0)<0.00001
            break
        end
        R0=R;
        t0=t;
        W0=W;
    end
end
[n m]=size(PC1);
PC1t=PC1*R'+ones(n,1)*t;
plot3(PC1t(:,1),PC1t(:,2),PC1t(:,3),'.b','MarkerSize',1);
hold on;
plot3(PC2(:,1),PC2(:,2),PC2(:,3),'.g','MarkerSize',1);
set(gca,'DataAspectRatio',[1 1 1]);
axis off

