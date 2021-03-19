import math
import time

from objloader import Obj
from pyrr import Matrix44
import moderngl_window as mglw
import moderngl
import numpy as np
from PIL import Image
aspect_ratio = 16 / 9
FLAG_DELETE=-100
debug=0

def set_camera( fov, eye, target):
    proj = Matrix44.perspective_projection(fov, aspect_ratio, 0.1, 1000.0)
    look = Matrix44.look_at(eye, target, (0.0, 0.0, 1.0))
    Mvp=(proj * look).astype('f4').tobytes()
    return Mvp

def plucker(a, b):
    l0 = a[0] * b[1] - b[0] * a[1]
    l1 = a[0] * b[2] - b[0] * a[2]
    l2 = a[0] - b[0]
    l3 = a[1] * b[2] - b[1] * a[2]
    l4 = a[2] - b[2]
    l5 = b[1] - a[1]
    return [l0, l1, l2, l3, l4, l5]

def sideOp(a, b):
    res = a[0] * b[4] + a[1] * b[5] + a[2] * b[3] + a[3] * b[2] + a[4] * b[0] + a[5] * b[1]
    return res

def compute_2D_intersection(T,L):
    p=[[0,0,0],[0,1,0],[0,0,1],[1,0,0]]
    x=p[1]
    edge=[0,0,0]
    vertex=[0,0,0]
    intersection=0
    for i in  range(4):
        T1=[T[0],T[1],p[i]]
        s1=  sideOp(plucker(L[0], L[1]), plucker(T1[0], T1[1]))
        s2 = sideOp(plucker(L[0], L[1]), plucker(T1[1], T1[2]))
        s3 = sideOp(plucker(L[0], L[1]), plucker(T1[0], T1[2]))
        if (s1==0 and s2==0 and s3==0) is False:
            x = p[i]
            print('----xx:',x)
            break
    for i in range(3):
        T1=[T[i],T[(i+1)%3],x]
        s2=sideOp(plucker(L[0],L[1]),plucker(T1[1],T1[2]))
        print('=--=:',L,T1)
        s3 = sideOp(plucker(L[0], L[1]), plucker(T1[2], T1[0]))

        if s2*s3<0:
            pass
        elif s2*s3>0:
            edge[i]+=1
            intersection+=1
        elif s2==0 and s3==0:
            edge[i] += 2
            intersection += 2
            break
        elif s2==0:
            vertex[int((i+1)-3*((i+1)/3))]+=1
            intersection+=1
        elif s3==0:
            vertex[i]+=1
            intersection+=1
    if intersection==0:
        if debug:
            print('In 2D,Line and Triangle do not intersect')
        return False
    elif debug==1:
        for i in range(3):
            if edge[i]==1 and debug==1:
                print('In 2D,The Line intersects the edge')
            if edge[i]==2 and debug==1:
                print('In 2D,The Line touches the edge')
                break
            for i in range(3):
                if vertex[i]!=0 and intersection<=3:
                    print('In 2D,The Line passes through the vertex')
    else:
        return True


class LoadOBJModel(mglw.WindowConfig):
    title = 'Hello World'
    gl_version = (3, 3)
    window_size = (1280, 720)
    aspect_ratio = 16 / 9
    resizable = True
    SetColor=(0.3,0.4,0.8)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prog = self.ctx.program(
            vertex_shader='''
                        #version 330

                        uniform mat4 Mvp;

                        in vec3 in_vert;
                        in vec3 in_norm;
                        in vec3 in_color;
                        in vec2 in_text;

                        out vec3 v_vert;
                        out vec3 v_norm;
                        out vec3 v_color;
                        out vec2 v_text;

                        void main() {
                            gl_Position = Mvp * vec4(in_vert, 1.0);
                            v_vert = in_vert;
                            v_norm = in_norm;
                            v_color = in_color;
                            v_text = in_text;
                        }
                    ''',
            fragment_shader='''
                        #version 330

                        uniform sampler2D Texture;
                        uniform int RenderMode;
                        uniform vec3 Color;
                        uniform vec3 Light;

                        in vec3 v_vert;
                        in vec3 v_norm;
                        in vec3 v_color;
                        in vec2 v_text;

                        out vec4 f_color;

                        void main() {
                            float lum = 0.2 + 0.8* abs(dot(normalize(Light - v_vert), normalize(v_norm)));

                            if (RenderMode == 0) {
                                f_color = vec4(Color, 0.4);
                            } else if (RenderMode == 1) {
                                f_color = vec4(Color * lum, 0.4);
                            } else if (RenderMode == 2) {
                                f_color = vec4(v_color * lum, 0.4);
                            } else if (RenderMode == 3) {
                                f_color = texture(Texture, v_text) * vec4(lum, lum, lum, 0.4);
                            }
                        }
                    ''',
        )
        self.mvp = self.prog['Mvp']
        self.light = self.prog['Light']
        self.renderMode=self.prog['RenderMode']
        self.color=self.prog['Color']


        obj = Obj.open(r'Recorde.obj')
        self.vbo = self.ctx.buffer(obj.pack('vx vy vz nx ny nz'))
        self.vao = self.ctx.vertex_array(self.prog, self.vbo, 'in_vert','in_norm')


    def render(self,time: float, frame_time: float):

        angle = time
        #print('angle:',angle)
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        camera_pos = (np.cos(angle)*12, np.sin(angle)*12, 1)

        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            camera_pos,
            (0.0, 0.0, 0.1),
            (0.0, 0.0, 1),
        )

        self.mvp.write((proj * lookat).astype('f4'))
        self.light.value = camera_pos
        self.color.value=self.SetColor
        self.renderMode.value=1
        self.vao.render()

    def setColor(self,Color):
            self.SetColor = Color
           # print('SetColor',SetColor)
    @classmethod
    def run(cls):
        mglw.run_window_config(cls)

class Polyhedron:
    def __init__(self):
        self.vertices=[]
        self.norms=[]
        self.faces=[]
        self.edges=[]
        self.BuildTime=0
        self.pName='polyhedron1'
    def IsSameFace(self, face1, face2):

        for i in range(3):
            if face1[i][0] != face2[i][0] or face1[i][1] != face2[i][1]:
                return False
        return True

    # def IsSameVert(self, vert1, vert2):
    #     for i in range(3):
    #         if vert1[i] != vert2[i]:
    #             return False
    #     return True

    def IsSameEdge(self, edge1, edge2):

        if (edge1[0] == edge2[0] and edge1[1] == edge2[1]) or (edge1[0] == edge2[1] and edge1[1] == edge2[0]):
            return True
        else:
            return False

    def CaculateNormal(self,vert):

        # for idx in vertices_index:
        #     verts.append(self.vertices[idx])
        vertices=np.array(vert)
        x1=vertices[0]-vertices[1]
        x2=vertices[0]-vertices[2]
        normal= np.cross(x1,x2)
        normal=normal / np.linalg.norm(normal)
        return normal

    def addVert(self,vert):
        self.vertices.append(vert)

    def FindSameFace(self,norm,fVert):
        for f in self.faces:
            pass
    def ExistSameEdge(self,newEdge):

        for edge in self.edges:
            if self.IsSameEdge(edge,newEdge):
                return False
        return True

    def addEdge(self,vert1,vert2):

        newEdge=[vert1,vert2]
        if self.ExistSameEdge(newEdge):
            self.edges.append(newEdge)
            return newEdge
        else:
            return None

    def FindOutterVert(self,faceVert): #Find a vert belong to polyhedron but not belong to current Face
        for v in self.vertices:
            if v!=faceVert[0] and v!=faceVert[1] and v!=faceVert[2]:
                return v

    def addFace(self,faceVert):
        Vindex=faceVert# indexes of faceVerts in self.vertices

        LenV=len(faceVert)
        #addEdge
        edges=[]
        for i in range(LenV):
            self.addEdge(faceVert[i],faceVert[(i+1)%LenV])
            edges.append([faceVert[i],faceVert[(i+1)%LenV]])
        #print('the num of edges of a face ',len(edges))


        OutterVert=self.FindOutterVert(faceVert)
        norm = self.CaculateNormal(faceVert)
        #make sure the norm is outter norm
        if len(self.vertices)>3:
            for i in range(3):# avoid norm.dot==0
                vec_out=np.array(OutterVert)-np.array(faceVert[i])

                Flag_outter=norm.dot(vec_out)
                if Flag_outter>0:
                    norm=-1*norm
                    break
                elif Flag_outter<0:
                    break
                    #norm=self.FindSameFace(norm)

        norm=norm.tolist()# force the type to be 'list'


        self.norms.append(norm)
        #Nindex = len(self.norms)-1

        self.faces.append([[faceVert[0],norm],[faceVert[1],norm],[faceVert[2],norm],edges])

    def initHedron(self, vertices):

        #self.vertices.extend(vertices)
        #self.Vnum+=4
        self.addVert(vertices[0])
        self.addVert(vertices[1])
        self.addVert(vertices[2])
        self.addVert(vertices[3])


        # fVert1=  [0,1,2]
        # fVert2 = [0,1,3]
        # fVert3 = [0,2,3]
        # fVert4 = [1,2,3]

        fVert1=  [vertices[0],vertices[1],vertices[2]]
        fVert2 = [vertices[0],vertices[1],vertices[3]]
        fVert3 = [vertices[0],vertices[2],vertices[3]]
        fVert4 = [vertices[1],vertices[2],vertices[3]]

        self.addFace(fVert1)
        self.addFace(fVert2)
        self.addFace(fVert3)
        self.addFace(fVert4)
    def IndexOfVert(self,vert):
        for idx,v in self.vertices:
            if v==vert:
                return idx+1

        print('Not found the vert in self.Vertices')
        return -1

    def IndexOfVert(self, vert):
        for idx, v in self.vertices:
            if v == vert:
                return idx + 1

        print('Not found the vert in self.Vertices')
        return -1


    def WriteObj(self):

        print('num of verts:',len(self.vertices))
        print('num of norms:',len(self.norms))
        print('num of faces:', len(self.faces))
        print('num of edges:', len(self.edges))

        with open('Recorde.obj','w+') as ff:
            for v in self.vertices:
                ff.write(f'v {v[0]} {v[1]} {v[2]}\n')
            for vn in self.norms:

                ff.write(f'vn {vn[0]} {vn[1]} {vn[2]}\n')

            ff.write(f'g {self.pName} \n')
            for f in self.faces:
                idxVs=[]
                idxN=self.norms.index(f[0][1])+1
                flag=0
                for i in range(3):
                    if f[i][0]  in self.vertices:
                        idxVs.append(self.vertices.index(f[i][0])+1)
                    else:
                        flag=1
                        print('******************************************************')
                        print('cannot find the vert:',f[i][0])
                        print('face:', f)
                        print('verts', self.vertices)
                        print('******************************************************')
                if flag==0:
                    ff.write(f'f {idxVs[0]}//{idxN} {idxVs[1]}//{idxN} {idxVs[2]}//{idxN}\n')

    def WriteObj_Add(self,name):
        print(' add num of verts:', len(self.vertices))
        print('add num of norms:', len(self.norms))
        print('add num of faces:', len(self.faces))
        print('add num of edges:', len(self.edges))

        obj=Obj.open('Recorde.obj')
        LenVert=len(obj.vert)
        LenNorm=len(obj.norm)
        with open('Recorde.obj', 'a+') as ff:
            for v in self.vertices:
                ff.write(f'v {v[0]} {v[1]} {v[2]}\n')
            for vn in self.norms:
                ff.write(f'vn {vn[0]} {vn[1]} {vn[2]}\n')

            ff.write(f'g {name}\n')
            for f in self.faces:
                idxVs = []
                idxN = self.norms.index(f[0][1]) + 1
                flag = 0
                for i in range(3):
                    if f[i][0] in self.vertices:
                         idxVs.append(self.vertices.index(f[i][0]) + 1)
                    else:
                        flag=1
                if flag==0:
                    ff.write(f'f {idxVs[0]+LenVert}//{idxN+LenNorm} {idxVs[1]+LenVert}//{idxN+LenNorm} {idxVs[2]+LenVert}//{idxN+LenNorm}\n')


    def IsVertInner(self,vert):


        for f in self.faces:
            vec1=np.array(vert)-np.array(f[0][0])
            norm=np.array(f[0][1])
            if vec1.dot(norm)>0:
                return False

        return True

    def IsvertInEdge(self,vert,edgelist):
        for edge in edgelist:
            if vert in edge:
                return True
        return False

    def SearchVertIndex(self,vertx):
        for idx,v in enumerate(self.vertices):
            if v[0]==vertx[0] and v[1]==vertx[1] and v[2]==vertx[2]:
                return idx
        return -1
    def FindRepeatEdges(self,edgelist):
        deleteEdges=[]
        for edgeOut in edgelist:
            count=0
            for edgeIn in edgelist:
                if self.IsSameEdge(edgeIn,edgeOut):
                    count+=1
                    if count==2:
                        break
            if count==2:
                deleteEdges.append(edgeOut)
        return  deleteEdges


    def NewFaceVertsList(self,FlagDelete,newVert):

        vertslist=[]
        edgelist=[]
        for idx in FlagDelete:
            verts= [self.faces[idx][0][0],self.faces[idx][1][0],self.faces[idx][2][0]]
            for edge in self.faces[idx][3]:
                edgelist.append(edge)
            for v in verts:
                 if v not in vertslist:
                     vertslist.append(v)
        #delete Edges
        deleteEdges=self.FindRepeatEdges(edgelist)
        for dEdge in deleteEdges:
            edgelist.remove(dEdge)
            if dEdge in self.edges:
                self.DeleteEdge(dEdge)
                if debug==1:
                    print('_______Delete edge', dEdge)

        #delete inner vert
        for v in vertslist:
            if self.IsvertInEdge(v,edgelist) is False:
                self.DeleteVert(v)
                if debug == 1:
                    print('_______Delete vert', v)
                vertslist.remove(v)
        #delete faces
        self.DeleteFace(FlagDelete)



        #print('edgelist:',edgelist)
        NewFaceVerts=[]
       # idx_NewVert=self.SearchVertIndex(newVert)
        for edge in edgelist:
            NewFaceVerts.append([newVert,edge[0],edge[1]])

        return NewFaceVerts
    def DeleteVert(self,vert):
        self.vertices.remove(vert)

    def DeleteEdge(self,edge):
        if edge not in self.edges:
            reverse_edge=[edge[1],edge[0]]
            self.edges.remove(reverse_edge)
        else:
            self.edges.remove(edge)
    def DeleteNorm(self,norm):
        self.norms.remove(norm)

    def DeleteFace(self,faces_idx):

        for i,idx in enumerate(faces_idx):
            norm=self.faces[idx-i][0][1]
            self.DeleteNorm(norm)
            # self.faces[idx - i][0][1]-1
            # del self.norms[norm_idx-i]
            del self.faces[idx-i]
            if debug==1:
                print('_______Delete face:',idx)




    def TryAddOneVert(self,newVert):

        FlagDelete_faces=[]
        if self.IsVertInner(newVert) is False:
            self.addVert(newVert)
            for idx,f in enumerate(self.faces):

                cur=0
                vec1=np.array(newVert)-np.array(f[cur][0])
                norm = np.array(f[0][1])
                visiable=vec1.dot(norm)
                if visiable>0:
                    FlagDelete_faces.append(idx)

            newFaceVert=self.NewFaceVertsList(FlagDelete_faces,newVert)

            #print('newFacevert',newFaceVert)
            for FaceVert in newFaceVert:
                self.addFace(FaceVert)

    def IsSameLine(self,newvert,verts):
        vec1=np.array(newvert)-np.array(verts[0])
        vec2=np.array(newvert)-np.array(verts[1])
        cross=np.linalg.norm(np.cross(vec1,vec2))
        if cross==0:
            return True
        else:
            return False

    def IsSamePlane(self,newvert,verts):
        vec1 = np.array(verts[2]) - np.array(verts[0])
        vec2 = np.array(verts[2]) - np.array(verts[1])

        vec3=np.array(newvert)-np.array(verts[1])
        norm=np.cross(vec1,vec2)

        if np.dot(norm,vec3)==0:
            return True
        else:
            return False

    def BuildHull(self,vertices):
        #choose first 4 points to build init polyhedron
        begin_time=time.time()
        initPoints=[]
        for v in vertices:
            Len=len(initPoints)
            if Len==0:
                initPoints.append(v)
            elif Len==1 and v not in initPoints:
                initPoints.append(v)
            elif Len==2 and v not in initPoints and self.IsSameLine(v,initPoints) is False:
                initPoints.append(v)
            elif Len==3 and v not in initPoints and self.IsSamePlane(v,initPoints) is False:
                initPoints.append(v)

        for v in initPoints:
            vertices.remove(v)

        self.initHedron(initPoints)

        for v in vertices:
            self.TryAddOneVert(v)
        self.BuildTime=time.time()-begin_time
        print('====Time cost:',self.BuildTime,'s ====')
        return initPoints,vertices



    def IsIntersectTriangleSegement1(self, edge, face):
        # 计算
        #reference https://members.loria.fr/SLazard/ARC-Visi3D/Pant-project/files/Line_Triangle.html
        faceVert=[face[0][0],face[1][0],face[2][0]]
        e1 = plucker(faceVert[1], faceVert[0])
        e2 = plucker(faceVert[2], faceVert[1])
        e3 = plucker(faceVert[0], faceVert[2])
        L = plucker(edge[0], edge[1])

        norm=face[0][1]
        vec1=np.array(edge[0])-np.array(faceVert[0])
        vec2 = np.array(edge[1]) - np.array(faceVert[0])
        SameSide=np.dot(norm,vec1)*np.dot(norm,vec2)




        s1 = sideOp(L, e1)
        s2 = sideOp(L, e2)
        s3 = sideOp(L, e3)

        if s1 == 0 and s2 == 0 and s3 == 0:
            #print("线和三角形共面")
            #print(faceVert,edge)
            return compute_2D_intersection(faceVert,edge)
        elif (s1 > 0 and s2 > 0 and s3 > 0) or (s1 < 0 and s2 < 0 and s3 < 0):
            #print("线穿过三角形")
            if SameSide<0:
                return True
        elif debug==1 and ((s1 == 0 and s2 * s3 > 0) or (s2 == 0 and s1 * s3 > 0) or (s3 == 0 and s1 * s2 > 0)):
            #print("线穿过三角形边缘")
            return True
        elif debug==1 and ((s1 == 0 and (s2 == 0)) or (s1 == 0 and (s3 == 0)) or (s2 == 0 and (s3 == 0))):
           # print("线穿过三角形顶点")
            return True
        else:
            return False

    def IsIntersectTriangleSegement2(self,edge,face):

        vecLine=np.array(edge[1])-np.array(edge[0])
        vecInFace1=np.array(face[1])-np.array(face[0])
        vecInFace2 = np.array(face[2][0]) - np.array(face[0][0])
        normFace=face[0][1]

        #Compute denominator d. If d <= 0, segment is parallel to or points
        #away from triangle, so exit early
        #d = 0.0 说明 qp 和 norm 垂直，说明三角形和 qp 平行。
	    #d < 0.0 说明 qp 和 norm 是钝角 说明是从三角形的背面 进入和三角形相交的
        d=np.dot(vecLine,normFace)
        if d<=0:# same side of
            return False


        #Compute intersection t value of pq with plane of triangle. A ray
        #intersects if 0 <= t. Segment intersects if 0 <= t <= 1. Delay
        #dividing by d until intersection has been found to pierce triangle
        vecLineToFace=np.array(edge[0])-np.array(edge[1])
        t=np.dot(vecLineToFace,normFace)
        if t<0:
            return False
        if t>d:
            return False

        #Compute barycentric coordinate components and test if within bound
        e=np.cross(vecLine,vecLineToFace)
        v=np.dot(vecInFace1,e)
        if v<0 or v>d:
            return False
        w=-1*np.dot(vecInFace2,e)
        if w<0 or v+w>d:
            return False

        #Segment/ray intersects triangle. Perform delayed division
        t=t/d

        return True


    def IsCollision(self,PolyHedron2):
        # 顶点在凸包内部
        for v in PolyHedron2.vertices:
            if self.IsVertInner(v):
                return True
        for v in self.vertices:
            if PolyHedron2.IsVertInner(v):
                return True
        #edge [[0.9501953125, 0.83984375, 0.58984375], [0.83984375, 0.740234375, 0.489990234375]]
        #face [[-0.56982421875, -0.7900390625, -0.409912109375], [0.489990234375, -0.830078125, -0.75], [[-0.7001953125, 0.389892578125, -0.3798828125]
        # 顶点在凸包外部，但边与另一凸包的面相交
        for edge in self.edges:
            for face in PolyHedron2.faces:
                if self.IsIntersectTriangleSegement1(edge,face):
                    return True
        # for edge in PolyHedron2.edges:
        #     for face in self.faces:
        #         if self.IsIntersectTriangleSegement1(edge, face):
        #             return True

        return False

def GenrateSet(n,offset=(-2,2)):
    acc=100
    defautSet = [[1, 0, 0],
                 [0, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1],
                 [1, 1, 0],
                 [1, 1, -1],
                 [0.2, 0.2, 0.2],
                 [1, 1, 1],
                 [5, -6, -4]]
    array_setB=np.random.randint(offset[0],offset[1],size=(n,3))
    array_setB=array_setB.astype(np.float16)/acc
    setB =  array_setB.tolist()

    return setB
    #f1=math.
def TwoInsectionHullSet2():#vertex is inner of polyhedron
    v1=[[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]]
    v2 =[[0.3, 0.3, 0.3], [0.3, 0.3, -2], [2, 1, 0], [0, 2, 0]]
    return v1,v2
def TwoInsectionHullSet1():# edge intersection with faces
    v1=[[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]]
    v2 =[[0.3, 0.3, 2], [0.3, 0.3, -2], [2, 1, 0], [0, 2, 0]]
    return v1,v2
def TwoInsectionHullSet4():# edge intersection with faces
    v1=[[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]]
    v2 =[[2, 0, 0], [2, 2, 2], [0, 2, 0], [0, 0, 2]]
    return v1,v2
def TwoInsectionHullSet3():#poylhedron is in anoter
    v1=[[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]]
    v2 =[[2, 0, 0], [0,-2, -1], [0, 2, 0], [0, 0, 3]]
    return v1,v2
def TwoRandomHulls(n):
    v1=GenrateSet(n,(-400,10))
    v2=GenrateSet(n,(0,400))
    return v1,v2
def CollisionDetection(set=5,vertnum=10):
    if set==1:
        v1,v2=TwoInsectionHullSet2()
    elif set==2:
        v1,v2=TwoInsectionHullSet1()
    elif set==3:
        v1,v2=TwoInsectionHullSet3()
    elif set==4:
        v1,v2=TwoInsectionHullSet4()
    else:
        v1,v2=TwoRandomHulls(vertnum)

    pA = Polyhedron()
    pB = Polyhedron()

    pA.BuildHull(v1)
    pA.WriteObj()

    Flag=pA.IsIntersectTriangleSegement1([[2,0,0],[0,0,2]],pA.faces[1])
    print('Flag=:',Flag)
    pB.BuildHull(v2)
    pB.WriteObj_Add('new')

    begintime = time.time()
    Collison = pB.IsCollision(pA)
    DetectionTime = time.time() - begintime

    print('=====================Detecion Time=============')
    print(DetectionTime)
    # Begin Render
    RenderModle = LoadOBJModel
    if Collison:
        RenderModle.setColor(RenderModle, (1, 0, 0))
    RenderModle.run()
    return DetectionTime
def GenerateAHull(choice,vertnum=10):
    defautSet_0 = [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]]
    defautSet_1 = [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
    defautSet_2 = [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],[2, 2, 0]]
    defautSet_3 = [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],[1, 1, -1],[0.2, 0.2, 0.2]]
    defautSet_4 = [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, -1], [0.2, 0.2, 0.2],[1, 1, 1]]
    defautSet_5 = [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, -1], [0.2, 0.2, 0.2],[1, 1, 1],[5, -6, -4] ]

    if choice==1:
        sets=defautSet_1
    elif choice==2:
        sets=defautSet_1
    elif choice==3:
        sets=defautSet_3
    elif choice==4:
        sets=defautSet_4
    elif choice==5:
        sets=defautSet_5
    else:
        sets=GenrateSet(vertnum,(-300,300))
    p=Polyhedron()
    p.BuildHull(sets)
    p.WriteObj()
    LoadOBJModel.run()

if __name__=='__main__':





   GenerateAHull(1)
  #   CollisionDetection(5,vertnum=40)

