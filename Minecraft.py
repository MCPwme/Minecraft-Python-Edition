import os
import math
import random
import ctypes
import OpenGL
import pyglet
import multiprocessing

from OpenGL.GL import *
from OpenGL.GLU import *
from pyglet import image
from pyglet.window import key,mouse
from pyglet.graphics import TextureGroup

OpenGL.ERROR_CHECKING=False

TICKS_PER_SEC=60
SECTOR_SIZE=16
VIEW_RADIUS=8
WORLD_RADIUS=48
HILL_COUNT=40
NUM_RENDER_WORKERS=4
WALKING_SPEED=5
FLYING_SPEED=15
GRAVITY=20.0
MAX_JUMP_HEIGHT=1.0
JUMP_SPEED=math.sqrt(2*GRAVITY*MAX_JUMP_HEIGHT)
TERMINAL_VELOCITY=50
PLAYER_HEIGHT=2
FACES=[(0,1,0),(0,-1,0),(-1,0,0),(1,0,0),(0,0,1),(0,0,-1)]

def normalize(position):
    x,y,z=position
    x,y,z=(int(round(x)),int(round(y)),int(round(z)))
    return (x,y,z)

def sectorize(position):
    x,y,z=normalize(position)
    return (x//SECTOR_SIZE,0,z//SECTOR_SIZE)

def cube_vertices(x,y,z,n):
    faces=[
        x-n,y+n,z-n,x-n,y+n,z+n,x+n,y+n,z+n,x+n,y+n,z-n,
        x-n,y-n,z-n,x+n,y-n,z-n,x+n,y-n,z+n,x-n,y-n,z+n,
        x-n,y-n,z-n,x-n,y-n,z+n,x-n,y+n,z+n,x-n,y+n,z-n,
        x+n,y-n,z+n,x+n,y-n,z-n,x+n,y+n,z-n,x+n,y+n,z+n,
        x-n,y-n,z+n,x+n,y-n,z+n,x+n,y+n,z+n,x-n,y+n,z+n,
        x+n,y-n,z-n,x-n,y-n,z-n,x-n,y+n,z-n,x+n,y+n,z-n,
    ]
    vertices=[]
    for face in range(6):
        start=face*12
        for index in (0,1,2,0,2,3):
            vertices.extend(faces[start+index*3:start+index*3+3])
    return vertices

def tex_coord(x,y,n=4):
    m=1.0/n
    dx=x*m
    dy=y*m
    return dx,dy,dx+m,dy,dx+m,dy+m,dx,dy+m

def tex_coords(top,bottom,side):
    top=tex_coord(*top)
    bottom=tex_coord(*bottom)
    side=tex_coord(*side)
    result=[]
    for face in (top,bottom,side,side,side,side):
        for index in (0,1,2,0,2,3):
            result.extend(face[index*2:index*2+2])
    return result

TEXTURE_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),'texture.png')
GRASS=tex_coords((1,0),(0,1),(0,0))
SAND=tex_coords((1,1),(1,1),(1,1))
BRICK=tex_coords((2,0),(2,0),(2,0))
STONE=tex_coords((2,1),(2,1),(2,1))

def _get_exposed_faces(position,world):
    x,y,z=position
    faces=[]
    for i,(dx,dy,dz) in enumerate(FACES):
        if (x+dx,y+dy,z+dz) not in world:
            faces.append(i)
    return faces

def _is_exposed(position,world):
    return bool(_get_exposed_faces(position,world))

def _nearby_sectors(center):
    if center is None:
        return set()
    x,_,z=center
    r=VIEW_RADIUS+1
    return {
        (x+dx,0,z+dz)
        for dx in range(-VIEW_RADIUS,VIEW_RADIUS+1)
        for dz in range(-VIEW_RADIUS,VIEW_RADIUS+1)
        if dx*dx+dz*dz <= r*r
    }

def _render_worker_loop(cmd_queue,result_queue):
    world={}
    sectors={}
    while True:
        cmd=cmd_queue.get()
        if cmd is None:
            break
        tag=cmd[0]
        if tag=='init':
            world=cmd[1]
            sectors={k: list(v) for k,v in cmd[2].items()}
        elif tag=='update':
            added=cmd[1]
            removed=cmd[2]
            for pos,tex in added:
                world[pos]=tex
                sec=sectorize(pos)
                sectors.setdefault(sec,[]).append(pos)
            for pos in removed:
                world.pop(pos,None)
                sec=sectorize(pos)
                lst=sectors.get(sec)
                if lst:
                    try:
                        lst.remove(pos)
                    except ValueError:
                        pass
        elif tag=='process':
            to_show,to_hide=cmd[1],cmd[2]
            results=[]
            for sector in to_show:
                for pos in sectors.get(sector,()):
                    if pos in world:
                        faces=_get_exposed_faces(pos,world)
                        if faces:
                            x,y,z=pos
                            verts=_make_cube_vertices_for_faces(x,y,z,faces)
                            tex=_extract_tex_for_faces(world[pos],faces)
                            results.append(('show',pos,tex,verts,faces))
            for sector in to_hide:
                for pos in sectors.get(sector,()):
                    results.append(('hide',pos))
            if results:
                result_queue.put(results)

_UNIT_CUBE=tuple(cube_vertices(0,0,0,0.5))

def _make_cube_vertices_for_faces(x,y,z,faces):
    uc=_UNIT_CUBE
    off=(x,y,z)
    result=[]
    for f in faces:
        start=f*18
        for i in range(start,start+18):
            result.append(uc[i]+off[i % 3])
    return tuple(result)

def _extract_tex_for_faces(tex,faces):
    result=[]
    for f in faces:
        start=f*12
        result.extend(tex[start:start+12])
    return result

class VertexMesh(object):
    __slots__=('count','mode','group','vertices','texcoords','_batch')
    def __init__(self,count,mode,group,vertices,texcoords,batch):
        self.count=count
        self.mode=mode
        self.group=group
        self.vertices=vertices
        self.texcoords=texcoords
        self._batch=batch

    def delete(self):
        if self._batch is not None:
            self._batch.remove(self)
            self._batch=None

class CompatBatch(object):
    def __init__(self):
        self.items=[]
        self._vbo_dirty=True
        self._vertex_count=0
        self._group=None
        self._vbo_verts=None
        self._vbo_tex=None
        self._vao=None
        self._vbo_mode=False
        self._all_verts=[]
        self._all_tex=[]
        self._mesh_offs={}
        self._pending_removals=0
        self._last_vc=0
        self._vbo_v_cap=0
        self._vbo_t_cap=0

    def add(self,count,mode,group,*data):
        vertex_data=None
        tex_data=None
        for fmt,values in data:
            clean_fmt=fmt.split('/')[0]
            if clean_fmt.startswith('v3'):
                vertex_data=tuple(values)
            elif clean_fmt.startswith('t2'):
                tex_data=tuple(values)
        mesh=VertexMesh(count,mode,group,vertex_data,tex_data,self)
        self.items.append(mesh)
        v_start=len(self._all_verts)
        t_start=len(self._all_tex)
        self._all_verts.extend(mesh.vertices)
        self._all_tex.extend(mesh.texcoords)
        self._mesh_offs[id(mesh)]=(v_start,t_start,count)
        self._vertex_count+=count
        self._vbo_dirty=True
        return mesh

    def remove(self,mesh):
        try:
            self.items.remove(mesh)
        except ValueError:
            return
        self._vertex_count-=mesh.count
        info=self._mesh_offs.pop(id(mesh),None)
        if info is not None:
            v_start,t_start,count=info
            del self._all_verts[v_start:v_start+count*3]
            del self._all_tex[t_start:t_start+count*2]
            self._pending_removals+=1
        self._vbo_dirty=True

    def _ensure_vbo_backend(self):
        if self._vbo_mode:
            return
        try:
            from OpenGL.raw.GL.ARB.vertex_array_object import (
                glGenVertexArrays,glBindVertexArray,glDeleteVertexArrays)
            self._glGenVertexArrays=glGenVertexArrays
            self._glBindVertexArray=glBindVertexArray
            self._glDeleteVertexArrays=glDeleteVertexArrays
            self._vbo_mode='vao'
        except (ImportError,RuntimeError):
            self._vbo_mode='vbo'

    def _rebuild_offsets(self):
        self._mesh_offs.clear()
        v=t=0
        for mesh in self.items:
            c=mesh.count
            self._mesh_offs[id(mesh)]=(v,t,c)
            v+=c*3
            t+=c*2

    def _init_vao(self,ZERO):
        try:
            vao=ctypes.c_uint()
            self._glGenVertexArrays(1,ctypes.byref(vao))
            self._glBindVertexArray(vao)
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER,self._vbo_verts)
            glVertexPointer(3,GL_FLOAT,0,ZERO)
            glBindBuffer(GL_ARRAY_BUFFER,self._vbo_tex)
            glTexCoordPointer(2,GL_FLOAT,0,ZERO)
            self._vao=vao
            self._glBindVertexArray(0)
        except Exception:
            self._vbo_mode='vbo'

    def _full_vbo_upload(self):
        while glGetError()!=GL_NO_ERROR:
            pass
        nv=len(self._all_verts)
        nt=len(self._all_tex)
        cap_v=max(nv*2,1024)
        cap_t=max(nt*2,1024)
        ZERO=ctypes.c_void_p(0)
        try:
            if self._vbo_verts is None:
                self._vbo_verts=glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER,self._vbo_verts)
            arr_v=(GLfloat*cap_v)(*self._all_verts + [0.0] * (cap_v - nv))
            glBufferData(GL_ARRAY_BUFFER,ctypes.sizeof(arr_v),arr_v,GL_STATIC_DRAW)
            if self._vbo_tex is None:
                self._vbo_tex=glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER,self._vbo_tex)
            arr_t=(GLfloat*cap_t)(*self._all_tex + [0.0] * (cap_t - nt))
            glBufferData(GL_ARRAY_BUFFER,ctypes.sizeof(arr_t),arr_t,GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER,0)
            if glGetError()!=GL_NO_ERROR:
                raise RuntimeError('glBufferData')
            self._vbo_v_cap=cap_v
            self._vbo_t_cap=cap_t
            self._last_vc=self._vertex_count
            if self._vbo_mode=='vao' and self._vao is None:
                self._init_vao(ZERO)
            if glGetError()!=GL_NO_ERROR:
                raise RuntimeError('vao')
        except Exception:
            self._cleanup_gpu_resources()
            self._vbo_mode=False
            self._vbo_v_cap=self._vbo_t_cap=0

    def _incr_vbo_upload(self):
        old_count=self._last_vc
        new_verts=self._all_verts[old_count*3:]
        new_tex=self._all_tex[old_count*2:]
        nv=len(new_verts)
        nt=len(new_tex)
        if nv>self._vbo_v_cap-old_count*3 or nt>self._vbo_t_cap-old_count*2:
            self._full_vbo_upload()
            return
        while glGetError()!=GL_NO_ERROR:
            pass
        try:
            glBindBuffer(GL_ARRAY_BUFFER,self._vbo_verts)
            arr_v=(GLfloat*nv)(*new_verts)
            glBufferSubData(GL_ARRAY_BUFFER,old_count*3*4,ctypes.sizeof(arr_v),arr_v)
            glBindBuffer(GL_ARRAY_BUFFER,self._vbo_tex)
            arr_t=(GLfloat*nt)(*new_tex)
            glBufferSubData(GL_ARRAY_BUFFER,old_count*2*4,ctypes.sizeof(arr_t),arr_t)
            glBindBuffer(GL_ARRAY_BUFFER,0)
            if glGetError()!=GL_NO_ERROR:
                raise RuntimeError('glBufferSubData')
            self._last_vc=self._vertex_count
        except Exception:
            self._cleanup_gpu_resources()
            self._vbo_mode=False
            self._vbo_v_cap=self._vbo_t_cap=0

    def _upload_to_gpu(self):
        self._vbo_dirty=False
        self._ensure_vbo_backend()
        if not self._vbo_mode:
            return
        if not self.items:
            self._cleanup_gpu_resources()
            self._vbo_v_cap=self._vbo_t_cap=self._last_vc=0
            self._pending_removals=0
            return
        self._group=self.items[0].group
        if self._pending_removals>0:
            self._rebuild_offsets()
            self._pending_removals=0
            self._full_vbo_upload()
        elif self._vertex_count!=self._last_vc:
            self._incr_vbo_upload()

    def _cleanup_gpu_resources(self):
        if self._vao is not None:
            try:
                self._glDeleteVertexArrays(1,ctypes.byref(self._vao))
            except Exception:
                pass
            self._vao=None
        if self._vbo_verts is not None:
            try:
                glDeleteBuffers(1,[self._vbo_verts])
            except Exception:
                pass
            self._vbo_verts=None
        if self._vbo_tex is not None:
            try:
                glDeleteBuffers(1,[self._vbo_tex])
            except Exception:
                pass
            self._vbo_tex=None

    def draw(self):
        if self._vertex_count==0:
            return
        group=self._group
        if group is None and self.items:
            group=self.items[0].group
        if group is not None:
            group.set_state()
        glEnable(GL_TEXTURE_2D)
        if self._vbo_dirty:
            self._upload_to_gpu()
        if self._vbo_mode and self._vbo_verts is not None:
            if self._vbo_mode=='vao' and self._vao is not None:
                try:
                    self._glBindVertexArray(self._vao)
                    glDrawArrays(GL_TRIANGLES,0,self._vertex_count)
                    self._glBindVertexArray(0)
                    if glGetError()==GL_NO_ERROR:
                        if group is not None:
                            group.unset_state()
                        return
                except Exception:
                    pass
                self._vbo_mode='vbo'
            if self._vbo_mode=='vbo':
                try:
                    ZERO=ctypes.c_void_p(0)
                    glEnableClientState(GL_VERTEX_ARRAY)
                    glEnableClientState(GL_TEXTURE_COORD_ARRAY)
                    glBindBuffer(GL_ARRAY_BUFFER,self._vbo_verts)
                    glVertexPointer(3,GL_FLOAT,0,ZERO)
                    glBindBuffer(GL_ARRAY_BUFFER,self._vbo_tex)
                    glTexCoordPointer(2,GL_FLOAT,0,ZERO)
                    glDrawArrays(GL_TRIANGLES,0,self._vertex_count)
                    glDisableClientState(GL_VERTEX_ARRAY)
                    glDisableClientState(GL_TEXTURE_COORD_ARRAY)
                    glBindBuffer(GL_ARRAY_BUFFER,0)
                    if glGetError()==GL_NO_ERROR:
                        if group is not None:
                            group.unset_state()
                        return
                except Exception:
                    pass
                self._cleanup_gpu_resources()
                self._vbo_mode=False
        verts=self._all_verts
        tex=self._all_tex
        n=self._vertex_count
        glBegin(GL_TRIANGLES)
        for i in range(n):
            vi=i*3
            ti=i*2
            glTexCoord2f(tex[ti],tex[ti+1])
            glVertex3f(verts[vi],verts[vi+1],verts[vi+2])
        glEnd()
        if group is not None:
            group.unset_state()

class Model(object):
    def __init__(self):
        self.batch=CompatBatch()
        self.group=TextureGroup(image.load(TEXTURE_PATH).get_texture())
        self.world={}
        self.shown={}
        self.mesh={}
        self.sectors={}
        self.render_queue=[]
        self._pending_world_add=[]
        self._pending_world_remove=[]
        self._result_queue=multiprocessing.Queue()
        self._cmd_queues=[]
        self._workers=[]
        for _ in range(NUM_RENDER_WORKERS):
            q=multiprocessing.Queue()
            self._cmd_queues.append(q)
            p=multiprocessing.Process(
                target=_render_worker_loop,
                args=(q,self._result_queue),
                daemon=True,
            )
            p.start()
            self._workers.append(p)
        self._initialize()

    def _initialize(self):
        radius=WORLD_RADIUS
        ground=1
        for x in range(-radius,radius+1):
            for z in range(-radius,radius+1):
                self.add_block((x,ground-2,z),GRASS,immediate=False)
                self.add_block((x,ground-3,z),STONE,immediate=False)
                if x in (-radius,radius) or z in (-radius,radius):
                    for height in range(-2,3):
                        self.add_block((x,ground+height,z),STONE,immediate=False)
        hill_range=radius-10
        for _ in range(HILL_COUNT):
            cx=random.randint(-hill_range,hill_range)
            cz=random.randint(-hill_range,hill_range)
            h=random.randint(1,6)
            r=random.randint(4,8)
            texture=random.choice([GRASS,SAND,BRICK])
            for y in range(-1,-1+h):
                for x in range(cx-r,cx+r+1):
                    for z in range(cz-r,cz+r+1):
                        if (x-cx) ** 2+(z-cz) ** 2 > (r+1) ** 2:
                            continue
                        if x ** 2+z ** 2 < 5 ** 2:
                            continue
                        self.add_block((x,y,z),texture,immediate=False)
                r -= 1
        self._sync_worker_full()

    def _sync_worker_full(self):
        data=(dict(self.world),{k: list(v) for k,v in self.sectors.items()})
        for q in self._cmd_queues:
            q.put(('init',data[0],data[1]))

    def _sync_worker_delta(self):
        if not self._pending_world_add and not self._pending_world_remove:
            return
        added=list(self._pending_world_add)
        removed=list(self._pending_world_remove)
        self._pending_world_add.clear()
        self._pending_world_remove.clear()
        for q in self._cmd_queues:
            q.put(('update',added,removed))

    def hit_test(self,position,vector,max_distance=8):
        m=8
        x,y,z=position
        dx,dy,dz=vector
        previous=None
        world=self.world
        for _ in range(max_distance*m):
            key=normalize((x,y,z))
            if key!=previous and key in world:
                return key,previous
            previous=key
            x,y,z=x+dx/m,y+dy/m,z+dz/m
        return None,None

    def exposed(self,position):
        return _is_exposed(position,self.world)

    def add_block(self,position,texture,immediate=True):
        if position in self.world:
            self.remove_block(position,immediate)
        self.world[position]=texture
        self.sectors.setdefault(sectorize(position),[]).append(position)
        self._pending_world_add.append((position,texture))
        if immediate:
            if self.exposed(position):
                self.show_block(position)
            self.check_neighbors(position)

    def remove_block(self,position,immediate=True):
        if position not in self.world:
            return
        del self.world[position]
        sec=sectorize(position)
        lst=self.sectors.get(sec)
        if lst is not None:
            try:
                lst.remove(position)
            except ValueError:
                pass
            if not lst:
                del self.sectors[sec]
        self._pending_world_remove.append(position)
        if immediate:
            if position in self.shown:
                self.hide_block(position)
            self.check_neighbors(position)

    def check_neighbors(self,position):
        x,y,z=position
        world=self.world
        shown=self.shown
        for dx,dy,dz in FACES:
            key=(x+dx,y+dy,z+dz)
            if key not in world:
                continue
            if _is_exposed(key,world):
                self.show_block(key)
            else:
                if key in shown:
                    self.hide_block(key)

    def show_block(self,position):
        texture=self.world[position]
        self.shown[position]=texture
        self.render_queue.append(('show',position,texture))

    def _show_block_mesh(self,position,texture):
        faces=_get_exposed_faces(position,self.world)
        if not faces:
            self.shown.pop(position,None)
            self._hide_block_mesh(position)
            return
        x,y,z=position
        verts=_make_cube_vertices_for_faces(x,y,z,faces)
        tex=_extract_tex_for_faces(texture,faces)
        vertex_count=len(faces)*6
        old=self.mesh.pop(position,None)
        if old is not None:
            old.delete()
        self.mesh[position]=self.batch.add(vertex_count,GL_TRIANGLES,
            self.group,
            ('v3f',verts),
            ('t2f',tex))

    def hide_block(self,position):
        self.shown.pop(position,None)
        self.render_queue.append(('hide',position))

    def _hide_block_mesh(self,position):
        item=self.mesh.pop(position,None)
        if item is not None:
            item.delete()

    def show_sector(self,sector):
        shown=self.shown
        world=self.world
        for position in self.sectors.get(sector,()):
            if position not in shown and _is_exposed(position,world):
                self.show_block(position)

    def hide_sector(self,sector):
        shown=self.shown
        for position in self.sectors.get(sector,()):
            if position in shown:
                self.hide_block(position)

    def change_sectors(self,before,after):
        old=_nearby_sectors(before)
        new=_nearby_sectors(after)
        show=sorted(new-old)
        hide=sorted(old-new)

        for sector in hide:
            self.hide_sector(sector)
        for sector in show:
            self.show_sector(sector)

        if show or hide:
            self._sync_worker_delta()
            nw=NUM_RENDER_WORKERS
            buckets=[
                (frozenset(show[i::nw]),frozenset(hide[i::nw]))
                for i in range(nw)
            ]
            for i,(show_set,hide_set) in enumerate(buckets):
                if show_set or hide_set:
                    self._cmd_queues[i].put(('process',show_set,hide_set))

    def _process_worker_results(self):
        while True:
            try:
                results=self._result_queue.get_nowait()
            except Exception:
                break
            for cmd in results:
                if cmd[0]=='show':
                    position,tex,verts,faces=cmd[1],cmd[2],cmd[3],cmd[4]
                    if position in self.shown:
                        continue
                    vertex_count=len(faces)*6
                    self.shown[position]=tex
                    old=self.mesh.pop(position,None)
                    if old is not None:
                        old.delete()
                    self.mesh[position]=self.batch.add(vertex_count,
                        GL_TRIANGLES,self.group,
                        ('v3f',verts),
                        ('t2f',tex))
                elif cmd[0]=='hide':
                    position=cmd[1]
                    self.shown.pop(position,None)
                    self._hide_block_mesh(position)

    def process_render_queue(self):
        if not self.render_queue:
            return
        queue=self.render_queue
        self.render_queue=[]
        for cmd in queue:
            if cmd[0]=='show':
                self._show_block_mesh(cmd[1],cmd[2])
            elif cmd[0]=='hide':
                self._hide_block_mesh(cmd[1])

    def shutdown(self):
        for q in self._cmd_queues:
            q.put(None)
        for p in self._workers:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()

class Window(pyglet.window.Window):
    def __init__(self,*args,**kwargs):
        super(Window,self).__init__(*args,**kwargs)
        self.exclusive=False
        self.flying=False
        self.strafe=[0,0]
        self.position=(0,0,0)
        self.rotation=(0,0)
        self.sector=None
        self.reticle=None
        self.dy=0
        self.inventory=[BRICK,GRASS,SAND]
        self.block=self.inventory[0]
        self.num_keys=[
            key._1,key._2,key._3,key._4,key._5,
            key._6,key._7,key._8,key._9,key._0]
        self.model=Model()
        self.sector=sectorize(self.position)
        self.model.change_sectors(None,self.sector)
        self.label=pyglet.text.Label('',font_name='Arial',font_size=18,
            x=10,y=self.height-10,anchor_x='left',anchor_y='top',
            color=(0,0,0,255))
        pyglet.clock.schedule_interval(self.update,1.0/TICKS_PER_SEC)

    def close(self):
        self.model.shutdown()
        super(Window,self).close()

    def set_exclusive_mouse(self,exclusive):
        super(Window,self).set_exclusive_mouse(exclusive)
        self.exclusive=exclusive

    def get_sight_vector(self):
        x,y=self.rotation
        m=math.cos(math.radians(y))
        dy=math.sin(math.radians(y))
        dx=math.cos(math.radians(x-90))*m
        dz=math.sin(math.radians(x-90))*m
        return (dx,dy,dz)

    def get_motion_vector(self):
        strafe=self.strafe
        if not any(strafe):
            return 0.0,0.0,0.0
        flying=self.flying
        rot_yaw=self.rotation[0]
        angle=math.radians(rot_yaw+math.degrees(math.atan2(*strafe)))
        if not flying:
            return math.cos(angle),0.0,math.sin(angle)
        pitch=math.radians(self.rotation[1])
        h=math.cos(pitch)
        v=math.sin(pitch)
        if strafe[1]:
            h,v=1.0,0.0
        if strafe[0] > 0:
            v *= -1
        return (math.cos(angle)*h,v,math.sin(angle)*h)

    def update(self,dt):
        self.model._process_worker_results()
        self.model.process_render_queue()
        dt=min(dt,0.2)
        steps=8
        sub_dt=dt/steps
        for _ in range(steps):
            self._update(sub_dt)

    def _update(self,dt):
        speed=FLYING_SPEED if self.flying else WALKING_SPEED
        distance=dt*speed
        dx,dy,dz=self.get_motion_vector()
        dx,dy,dz=dx*distance,dy*distance,dz*distance
        if not self.flying:
            self.dy -= dt*GRAVITY
            self.dy=max(self.dy,-TERMINAL_VELOCITY)
            dy += self.dy*dt
        x,y,z=self.position
        x,y,z=self._collide((x+dx,y+dy,z+dz),PLAYER_HEIGHT)
        self.position=(x,y,z)
        new_sector=sectorize(self.position)
        if new_sector!=self.sector:
            old=self.sector
            self.sector=new_sector
            self.model.change_sectors(old,new_sector)

    def _collide(self,position,height):
        pad=0.25
        result=list(position)
        block=normalize(position)
        world=self.model.world
        for face in FACES:
            for axis in range(3):
                if not face[axis]:
                    continue
                overlap=(result[axis]-block[axis])*face[axis]
                if overlap < pad:
                    continue
                for offset in range(height):
                    neighbor=list(block)
                    neighbor[1] -= offset
                    neighbor[axis] += face[axis]
                    if tuple(neighbor) not in world:
                        continue
                    result[axis] -= (overlap-pad)*face[axis]
                    if face[1]:
                        self.dy=0
                    break
        return tuple(result)

    def on_mouse_press(self,x,y,button,modifiers):
        if self.exclusive:
            vector=self.get_sight_vector()
            block,previous=self.model.hit_test(self.position,vector)
            if (button==mouse.RIGHT) or \
                    ((button==mouse.LEFT) and (modifiers & key.MOD_CTRL)):
                if previous:
                    self.model.add_block(previous,self.block)
            elif button==mouse.LEFT and block:
                texture=self.model.world.get(block)
                if texture and texture!=STONE:
                    self.model.remove_block(block)
        else:
            self.set_exclusive_mouse(True)

    def on_mouse_motion(self,x,y,dx,dy):
        if self.exclusive:
            m=0.15
            rot_x,rot_y=self.rotation
            rot_x += dx*m
            rot_y += dy*m
            rot_y=max(-90,min(90,rot_y))
            self.rotation=(rot_x,rot_y)

    def on_mouse_leave(self,x,y):
        if self.exclusive:
            self.set_exclusive_mouse(False)

    def on_deactivate(self):
        self.set_exclusive_mouse(False)
        self.strafe[:]=0,0

    def on_key_press(self,symbol,modifiers):
        if symbol==key.W:
            self.strafe[0] -= 1
        elif symbol==key.S:
            self.strafe[0] += 1
        elif symbol==key.A:
            self.strafe[1] -= 1
        elif symbol==key.D:
            self.strafe[1] += 1
        elif symbol==key.SPACE:
            if self.dy==0:
                self.dy=JUMP_SPEED
        elif symbol==key.ESCAPE:
            self.set_exclusive_mouse(False)
        elif symbol==key.TAB:
            self.flying=not self.flying
        elif symbol in self.num_keys:
            index=(symbol-self.num_keys[0]) % len(self.inventory)
            self.block=self.inventory[index]

    def on_key_release(self,symbol,modifiers):
        if symbol==key.W:
            self.strafe[0] += 1
        elif symbol==key.S:
            self.strafe[0] -= 1
        elif symbol==key.A:
            self.strafe[1] += 1
        elif symbol==key.D:
            self.strafe[1] -= 1

    def on_resize(self,width,height):
        self.label.y=height-10
        cx,cy=self.width//2,self.height//2
        n=10
        self.reticle=(
            cx-n,cy,0,
            cx+n,cy,0,
            cx,cy-n,0,
            cx,cy+n,0,
        )

    def set_2d(self):
        width,height=self.get_size()
        glDisable(GL_DEPTH_TEST)
        viewport=self.get_framebuffer_size()
        glViewport(0,0,max(1,viewport[0]),max(1,viewport[1]))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0,max(1,width),0,max(1,height),-1,1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def set_3d(self):
        width,height=self.get_size()
        glEnable(GL_DEPTH_TEST)
        viewport=self.get_framebuffer_size()
        glViewport(0,0,max(1,viewport[0]),max(1,viewport[1]))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(65.0,width/float(height),0.1,60.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        yaw,pitch=self.rotation
        px,py,pz=self.position
        glRotatef(-pitch,1.0,0.0,0.0)
        glRotatef(yaw,0.0,1.0,0.0)
        glTranslatef(-px,-(py+0.8),-pz)

    def on_draw(self):
        self.model._process_worker_results()
        self.model.process_render_queue()

        self.clear()
        self.set_3d()
        glColor3d(1,1,1)
        self.model.batch.draw()
        self.draw_focused_block()
        self.set_2d()
        self.draw_label()
        self.draw_reticle()

    def draw_focused_block(self):
        vector=self.get_sight_vector()
        block=self.model.hit_test(self.position,vector)[0]
        if block:
            x,y,z=block
            vertex_data=cube_vertices(x,y,z,0.51)
            glColor3d(0,0,0)
            glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)
            glBegin(GL_TRIANGLES)
            for i in range(0,len(vertex_data),3):
                glVertex3f(vertex_data[i],vertex_data[i+1],vertex_data[i+2])
            glEnd()
            glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)

    def draw_label(self):
        x,y,z=self.position
        self.label.text='%02d (%.2f,%.2f,%.2f) %d/%d' % (
            get_fps(),x,y,z,
            len(self.model.mesh),len(self.model.world))
        self.label.draw()

    def draw_reticle(self):
        if not self.reticle:
            return
        glColor3d(0,0,0)
        ret=self.reticle
        glBegin(GL_LINES)
        glVertex3f(float(ret[0]),float(ret[1]),0)
        glVertex3f(float(ret[3]),float(ret[4]),0)
        glVertex3f(float(ret[6]),float(ret[7]),0)
        glVertex3f(float(ret[9]),float(ret[10]),0)
        glEnd()

def get_fps():
    fps=getattr(pyglet.clock,'get_fps',None)
    if callable(fps):
        return fps()
    clock=pyglet.clock.get_default()
    freq=getattr(clock,'get_frequency',None)
    return freq() if callable(freq) else 0

def setup_fog():
    if not all(name in globals() for name in (
        'GL_FOG','GL_FOG_COLOR','GL_FOG_HINT','GL_DONT_CARE',
        'GL_FOG_MODE','GL_LINEAR','GL_FOG_START','GL_FOG_END'
    )):
        return
    try:
        glEnable(GL_FOG)
        glFogfv(GL_FOG_COLOR,(GLfloat*4)(0.5,0.69,1.0,1))
        glHint(GL_FOG_HINT,GL_DONT_CARE)
        glFogi(GL_FOG_MODE,GL_LINEAR)
        glFogf(GL_FOG_START,20.0)
        glFogf(GL_FOG_END,60.0)
    except Exception:
        pass

def setup():
    glClearColor(0.5,0.69,1.0,1)
    glDisable(GL_CULL_FACE)
    glEnable(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST)
    setup_fog()

def main():
    multiprocessing.freeze_support()
    width,height=1280,720
    screen=pyglet.display.get_display().get_default_screen()
    left=(screen.width-width)//2
    top=(screen.height-height)//2
    config=pyglet.gl.Config(
        major_version=2,
        minor_version=1,
        depth_size=24,
        double_buffer=True,
        sample_buffers=0)
    window=Window(width=width,
        height=height,
        caption='Minecraft Python Edition v0.1.5',
        resizable=True,
        config=config)
    window.set_location(left,top)
    window.set_exclusive_mouse(True)
    setup()
    pyglet.app.run()

if __name__=='__main__':
    main()