#ifndef __CUDA__GL_RESOURCE__
#define __CUDA__GL_RESOURCE__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

class   CudaGlBuffer                // VBO
{
    cudaGraphicsResource* resource;
    VertexBuffer* buffer;
    GLenum                 target;

public:
    CudaGlBuffer(VertexBuffer* buf, GLenum theTarget, unsigned int flags = cudaGraphicsMapFlagsWriteDiscard)        // cudaGraphicsMapFlagsReadOnly
    {
        buffer = buf;
        target = theTarget;

        buffer->bind(target);
        cudaGraphicsGLRegisterBuffer(&resource, buffer->getId(), flags);
        buffer->unbind();
    }

    ~CudaGlBuffer()
    {
        cudaGraphicsUnregisterResource(resource);
    }

    bool    mapResource(cudaStream_t stream = 0)
    {
        return cudaGraphicsMapResources(1, &resource, stream) == cudaSuccess;
    }

    bool    unmapResource(cudaStream_t stream = 0)
    {
        return cudaGraphicsUnmapResources(1, &resource, stream) == cudaSuccess;
    }

    void* mappedPointer(size_t& numBytes) const
    {
        void* ptr;

        if (cudaGraphicsResourceGetMappedPointer(&ptr, &numBytes, resource) != cudaSuccess)
            return NULL;

        return ptr;
    }

    GLuint  getId() const
    {
        return buffer->getId();
    }

    GLenum getTarget() const
    {
        return target;
    }

    cudaGraphicsResource* getResource() const
    {
        return resource;
    }
};

class   CudaGlImage                 // texture or renderbuffer
{
    GLuint                 image;
    GLenum                 target;
    cudaGraphicsResource* resource;

public:
    CudaGlImage(GLuint theImage, GLenum theTarget, unsigned int  flags = cudaGraphicsMapFlagsWriteDiscard)   // cudaGraphicsMapFlagsReadOnly, cudaGraphicsMapFlagsNone
    {
        image = theImage;
        target = theTarget;
        cudaGraphicsGLRegisterImage(&resource, image, target, flags);
    }

    ~CudaGlImage()
    {
        cudaGraphicsUnregisterResource(resource);
    }

    bool    mapResource(cudaStream_t stream = 0)
    {
        return cudaGraphicsMapResources(1, &resource, stream) == cudaSuccess;
    }

    bool    unmapResource(cudaStream_t stream = 0)
    {
        return cudaGraphicsUnmapResources(1, &resource, stream) == cudaSuccess;
    }

    cudaArray* mappedArray(unsigned int index = 0, unsigned int mipLevel = 0) const
    {
        cudaArray* array;

        if (cudaGraphicsSubResourceGetMappedArray(&array, resource, index, mipLevel) != cudaSuccess)
            return NULL;

        return array;
    }

    GLuint  getImage() const
    {
        return image;
    }

    GLenum getTarget() const
    {
        return target;
    }
};

#endif