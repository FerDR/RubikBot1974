# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:45:44 2019

@author: Fer
"""
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import facebook
from pathlib import Path
import os
import skimage.measure as skm
import matplotlib.pyplot as plt


class Cube:
    def __init__(self,status = ''):
        self.rojo = [255,0,0]
        self.naranja = [255,127,39]
        self.verde = [0,255,0]
        self.azul = [0,0,255]
        self.amarillo = [255,242,0]
        self.blanco = [240,240,240]
        self.colors = [self.rojo,self.naranja,self.verde,
                       self.azul,self.amarillo,self.blanco]
        self.state = self.geninitialstate()
        if status != 'new':
            self.shuffle()
        self.label, self.props = self.genlabelprops()
        
    def issolved(self):
        up = [1,3,5,7,9,11,15,17,23]
        front = [19,25,45,29,33,49,37,41,53]
        right = [21,35,13,27,39,31,47,51,43]
        down = [0,2,4,6,8,10,14,16,22]
        back = [12,20,34,30,26,46,42,38,50]
        left = [18,24,44,28,32,36,40,48,52]
        colarrs = np.array([up,front,right,down,back,left])
        for colarr in colarrs:
            a = list(self.state[colarr[0]])
            for ic in colarr:
                b = list(self.state[ic])
                if a != b:
                    return False
        return True
       
    def geninitialstate(self):
        state = list(np.zeros(54))
        blancos = [1,3,5,7,9,11,15,17,23]
        azules = [19,25,45,29,33,49,37,41,53]
        naranjas = [21,35,13,27,39,31,47,51,43]
        amarillos = [0,2,4,6,8,10,14,16,22]
        verdes = [12,20,34,30,26,46,42,38,50]
        rojos = [18,24,44,28,32,36,40,48,52]
        colarrs = np.array([blancos,azules,naranjas,amarillos,verdes,rojos])
        colnames = [self.blanco,self.azul,self.naranja,
                    self.amarillo,self.verde,self.rojo]
        for it in range(len(state)):
            state[it] = colnames[np.argwhere(colarrs==it)[0][0]]
        return state
    
    def shuffle(self):
        for i in range(50):
            rot = np.random.randint(0,9)
            self.rotate(rot)
        
    def plot(self,savename = '', show = True, inline = True):
        matr = np.zeros(np.shape(self.label))
        matg = np.zeros(np.shape(self.label))
        matb = np.zeros(np.shape(self.label))
        for i in range(54):
            matr+=self.state[i][0]*np.array(self.getsquarematrix(i))
            matg+=self.state[i][1]*np.array(self.getsquarematrix(i))
            matb+=self.state[i][2]*np.array(self.getsquarematrix(i))
        r = Image.fromarray(matr).convert('L')
        b = Image.fromarray(matb).convert('L')
        g = Image.fromarray(matg).convert('L')
        im = Image.merge("RGB", (r, g, b))
        if show:
            if inline:
                return im
            else:
                im.show()
        if savename:
            im.save(savename)
    
    def rotate(self,face,way=''):
        up = [[23,11],[17,5],[11,1],[5,3],[1,7],[3,15],[7,23],[15,17],[13,29],
              [21,25],[27,19],[19,52],[25,48],[29,44],[44,50],[48,46],[52,42],
              [50,13],[46,21],[42,27]] 
        down = [[22,6],[14,2],[6,0],[2,4],[0,10],[4,16],[10,22],[16,14],
                [43,53],[47,49],[51,45],[45,28],[49,24],[53,18],[28,12],
                [24,20],[18,26],[12,51],[20,47],[26,43]]   
        #towards user
        left = [[1,19],[5,33],[11,45],[45,22],[33,16],[19,10],[50,11],
                [38,5],[26,1],[10,26],[16,38],[22,50],[18,28],[24,40],
                [28,52],[40,48],[52,44],[48,32],[44,18],[32,24]]
        back = [[1,13],[3,31],[7,43],[43,22],[31,14],[13,6],[52,7],[40,3],
                [28,1],[6,28],[14,40],[22,52],[12,26],[20,38],[26,50],
                [38,46],[50,42],[46,30],[42,12],[30,20]]
        front = [[11,27],[17,39],[23,51],[19,29],[25,41],[29,53],[41,49],
                 [53,45],[49,33],[45,19],[33,25],[44,23],[32,17],[18,11],
                 [51,10],[39,4],[27,0],[0,18],[4,32],[10,44]]
        right = [[7,29],[15,41],[23,53],[13,27],[21,39],[27,51],[39,47],
                 [51,43],[47,31],[43,13],[31,21],[53,6],[41,2],[29,0],
                 [42,23],[30,15],[12,7],[0,12],[2,30],[6,42]]
        #clockwise
        equatorial = [[31,41],[35,37],[39,33],[30,39],[34,35],[38,31],[33,40],
                      [37,36],[41,32],[32,38],[36,34],[40,30]]
        #as in left or right of the frontal faces, towards user
        middle = [[3,25],[9,37],[17,49],[20,3],[34,9],[46,17],[49,14],
                  [37,8],[25,4],[4,20],[8,34],[14,46]]
        standing = [[5,21],[9,35],[15,47],[48,15],[36,9],[24,5],[21,2],
                    [35,8],[47,16],[2,24],[8,36],[16,48]]     
        configs = [up,down,front,back,left,right,equatorial,middle,standing]
        confignames = ['up','down','front','back','left','right',
                       'equatorial','middle','standing']
        newstate = list(np.zeros(len(self.state)))
        if type(face) == str:
            face = confignames.index(face.lower())
        for switch in configs[face]:
            if way == 'reverse':
                newstate[switch[0]] = self.state[switch[1]]
            else:
                newstate[switch[1]] = self.state[switch[0]]
        for i in range(len(newstate)):
            if (type(newstate[i]) is int or type(newstate[i]) is float or 
            type(newstate[i]) is np.float64):
                newstate[i] = self.state[i]
        self.state = newstate
                    
    def continuous(self):
        while True:
            rot = input("Choose a rotation or write 'stop' to stop: ")
            if rot == 'stop':
                break
            rot = rot.split(',')
            for ir in rot:
                self.rotate(int(ir[0]),'reverse'*(ir[-1]=='r'))
            self.plot(inline=False)
                
    def mapper(self):
        k = 0
        for j in range(9):
            mat = np.zeros(np.shape(self.label))
            for i in range(6):
                mat+=(self.getsquarematrix(i+k)*(i+1))
            plt.imshow(mat,cmap='hsv')
            plt.pause(5)
            input("Press any key to continue")
            k+=6
    
    def genlabelprops(self):
        img = Image.open('rubik2.png')
        r0,g0,b0,what = img.split()
        r0 = np.array(r0)
        g0 = np.array(g0)
        b0 = np.array(b0)
        im1 =  r0>50 #semi-arbitrary color combo that yields desired result
        im2 = b0>200
        im = im1 + im2
        labeled, N_objects = skm.label( im, neighbors=4, return_num = True)
        objects = skm.regionprops(labeled) 
        props = [(object.area, object.label) for object in objects]
        #self.label = labeled
        #self.props = props
        return labeled,props
    
    def getsquarematrix(self,i):
        return self.label==self.props[i][1]

    def getsquarearea(self,i):
        return self.props[i][0]

    def savestate(self,filename='state'):
        np.save(filename,self.state)
        
    def loadstate(self,filename='state.npy'):
        self.state = np.load(filename)
        
    def getstatefromimage(self,image):
        img = Image.open(image)
        r0,g0,b0 = img.split()
        r0 = np.array(r0)
        g0 = np.array(g0)
        b0 = np.array(b0)
        for i in range(len(self.state)):
            r = r0*self.getsquarematrix(i)
            g = g0*self.getsquarematrix(i)
            b = b0*self.getsquarematrix(i)
            rm = np.mean(r)*np.size(r0)/self.getsquarearea(i)
            gm = np.mean(g)*np.size(r0)/self.getsquarearea(i)
            bm = np.mean(b)*np.size(r0)/self.getsquarearea(i)
            colmean = [rm,gm,bm]
            norm = 10000
            for col in self.colors:
                if np.linalg.norm(np.array(colmean)-np.array(col))<norm:
                    norm = np.linalg.norm(np.array(colmean)-np.array(col))
                    thiscol = col
            self.state[i] = thiscol
            
    def plotcornerhelp(self,savename=''):
        im = self.plot()
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype("arial.ttf",40)
        draw.text((5,100),'A',font=font,fill='magenta')
        draw.text((615,490),'A',font=font,fill='magenta')
        draw.text((230,-5),'B',font=font,fill='brown')
        draw.text((895,580),'B',font=font,fill='brown')
        draw.text((555,70),'C',font=font,fill='purple')
        draw.text((1155,480),'C',font=font,fill='purple')
        draw.text((5,490),'D',font=font,fill='aqua')
        draw.text((615,100),'D',font=font,fill='aqua')
        draw.text((345,580),'E',font=font,fill='olive')
        draw.text((820,-5),'E',font=font,fill='olive')
        draw.text((555,475),'F',font=font,fill='gray')
        draw.text((1155,70),'F',font=font,fill='gray')
        if im:
            im.save(savename)
        return im
#%%
def upload_comment(graph, post_id, message="", img_path=None):
    if img_path:
        post = graph.put_photo(image=open(img_path, 'rb'),
                               album_path='%s/comments' % (post_id),
                               message=message)
    else:
        post = graph.put_object(parent_object=post_id,
                                connection_name="comments",
                                message=message)
    return post
   
def upload_reply(graph, comment_id, message='',img_path=None):
    upload_comment(graph,comment_id,message,img_path)
 
def upload(message, access_token, img_path=None):
    graph = facebook.GraphAPI(access_token)
    if img_path:
        post = graph.put_photo(image=open(img_path, 'rb'),
                               message=message)
    else:
        post = graph.put_object(parent_object='me',
                                connection_name='feed',
                                message=message)
    return graph, post['post_id']

def getAccessToken(filename='access_token.txt'):
    return Path(filename).read_text().strip()

def getcomments(graph,post_id):
    comments = graph.get_connections(post_id,connection_name='comments')
    comments = comments['data']
    if comments:
        ids = []
        texts = []
        for comment in comments:
            ids.append(comment['from']['id'])
            texts.append(comment['message'])
        return ids,texts
    else:
        return [],[]

def filtercomments(ids,texts):
    if ids:
        ids_so_far = []
        filtered_texts=[]
        for it, text in enumerate(texts):
            if text[0]=='!':
                filtered_texts.append(text)
                ids_so_far.append(ids[it])
        indexes = np.unique(ids_so_far,return_index=True)[1]
        more_filtered_text = []       
        for i in indexes:
            more_filtered_text.append(filtered_texts[int(i)])
        return more_filtered_text
    else:
        return []

def getinputs(comments):
    if comments:
        inputs = []
        reverses = []
        letters = ['u','d','f','b','l','r','e','m','s']
        fullnames = ['up','down','front','back','left',
                     'right','equatorial','middle','standing']
        for comment in comments:
            comment = comment.lstrip('!')
            comment = comment.lower()
            comment = comment.split(' ')[0]
            lastletter = comment[-1]
            comment = comment.rstrip("!")
            for il,letter in enumerate(letters):
                if comment==letter:
                    inputs.append(il)
                    reverses.append(lastletter=='!')
            for ifn, fullname in enumerate(fullnames):
                if comment==fullname:
                    inputs.append(ifn)
                    reverses.append(lastletter=='!')
        output = []
        print(inputs,reverses)
        for i in range(len(inputs)):
            output.append([inputs[i],reverses[i]])
        return output
    else:
        return []

def findmostcommon(inputs):
    if inputs:
        mod_output = []
        for inp in inputs:
            mod_output.append(inp[0]+9*inp[1])
        m = np.bincount(mod_output).argmax()
        return [m-((m>8)*9),m>8]
    else:
        return [np.random.randint(0,8),np.random.rand()>0.5]
    
def main():
    if not os.path.isfile('state.npy'):
        cube = Cube()
        cube.savestate()
        cube.plotcornerhelp('initial.png')
        initial_message = ("Let's get started! Want to play? Check the "+
                           'pinned post, the description or the '+
                           'first comment for instructions.')
                      
        comment_message = ('To vote for a rotation, comment !<rotation>. '+
                           'You can see in the image the possible '+
                           'rotations. You can comment the full name or just '+
                           'the initial and it is not case-sensitive. For a '+
                           'rotation in the other way than the depicted in '+
                           'the picture simply add a ! at the end of your '+
                           'command. Only one command per person and the '+
                           'most voted one is chosen. For example !up! or '+
                           '!u! perform a reverse up-face rotation')
                             
        gr, p_id = upload(initial_message,getAccessToken(),'initial.png')
        c_id = upload_comment(gr,p_id,comment_message,'tutorial.png')['id']
        cube2 = Cube()
        cube2.plot(savename='random.png',show=False)
        upload_reply(gr,c_id,'Cubehaps','random.png')
        np.save('counter',[0])
        np.save('data',[gr,p_id])
        if cube.issolved():
            return False
        return True
   
    else:
        cube = Cube()
        cube.loadstate()
        i = np.load('counter.npy')[0]
        data = np.load('data.npy',allow_pickle=True)
        c0 = upload_comment(data[0],data[1],'Comments are no longer taken '+
                            'from this post')
        ids, texts = getcomments(data[0],data[1])
        coms = filtercomments(ids,texts)
        inps = getinputs(coms)
        print(inps)
        inp = findmostcommon(inps)
        print(inp)
        rotations = ['up','down','front','back','left',
                    'right','equatorial','middle','standing']
        cube.rotate(inp[0],'reverse'*int(inp[1]))
        cube.savestate()
        cube.plotcornerhelp('img'+str(i)+'.png')
        if cube.issolved():
            message = """Congratulations, the cube has been solved!
                         Stay tuned for the next run.
                         """
        else:
            message = ('A ' +int(inp[1])*'reverse '+ str(rotations[inp[0]]) +
                       ' rotation was made. '+
                       'Want to play? Check the '+
                       'pinned post, the description or the '+ 
                       'first comment for instructions.')
                         
        comment_message = ('To vote for a rotation, comment !<rotation>. '+
                           'You can see in the image the possible '+
                           'rotations. You can comment the full name or just '+
                           'the initial and it is not case-sensitive. For a '+
                           'rotation in the other way than the depicted in '+
                           'the picture simply add a ! at the end of your '+
                           'command. Only one command per person and the '+
                           'most voted one is chosen. For example !up! or '+
                           '!u! perform a reverse up-face rotation')
        
        gr,p_id = upload(message,getAccessToken(),'img'+str(i)+'.png')
        c_id = upload_comment(gr,p_id,comment_message,'tutorial.png')['id']
        cube2 = Cube()
        cube2.plot(savename='random.png',show=False)
        del cube2
        upload_reply(gr,c_id,'Cubehaps','random.png')
        if cube.issolved():
            return False
        i+=1
        np.save('data',[gr,p_id])
        np.save('counter',[i])
        return True