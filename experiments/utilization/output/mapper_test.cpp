#include "rpcmem_mapper.h"
#include "ggml-htp.h"
#include "ggml-htp-impl.h"
#include "ggml-backend-impl.h"
#include "dsprpc_interface.h"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <vector>

static ggml_backend_buffer_type buft{};
static std::map<void *,int> descriptors;
static std::vector<int> unmaps;
static int maps=0;
static void check(bool ok,const char *message) {
    if(!ok){std::fprintf(stderr,"FAIL: %s\n",message);std::exit(1);}
}
ggml_backend_htp_context *ggml_backend_htp_context::instance(){ std::abort(); }
bool ggml_backend_buft_is_rpcmem(ggml_backend_buffer_type_t type){return type==&buft;}
extern "C" int rpcmem_to_fd(void *base){return descriptors.at(base);}
extern "C" int fastrpc_mmap(int domain,int fd,void *,int,size_t,fastrpc_map_flags flags){
    check(domain==CDSP_DOMAIN_ID&&fd>0&&flags==FASTRPC_MAP_FD,"map API arguments");++maps;return 0;
}
extern "C" int fastrpc_munmap(int domain,int fd,void *,size_t){
    check(domain==CDSP_DOMAIN_ID&&fd>0,"unmap API arguments");unmaps.push_back(fd);return 0;
}
static void *base(ggml_backend_buffer_t buffer){return buffer->context;}
static ggml_backend_buffer_t make_buffer(void *address,int fd){
    descriptors[address]=fd;ggml_backend_buffer_i iface{};iface.get_base=base;
    return ggml_backend_buffer_init(&buft,iface,address,64);
}
int main(){
    char a[64],b[64],c[64];auto *ab=make_buffer(a,11),*bb=make_buffer(b,12),*cb=make_buffer(c,13);
    RpcMemMapper mapper(64,true);ggml_tensor destination{};RpcMemMapper::UnmapRequest found;
    destination.buffer=ab;mapper.validate(&destination);
    check(mapper.get_buffer_mapping(a,found)&&std::get<0>(found)==11,"active mapping visible");
    destination.buffer=bb;mapper.validate(&destination);
    check(mapper.get_pending_unmap_reqs().size()==1,"LRU eviction deferred");
    check(mapper.get_buffer_mapping(a,found)&&std::get<0>(found)==11,"evicted mapping still tracked");
    mapper.retire_buffer_mapping(a);
    check(unmaps.size()==1&&unmaps[0]==11,"pending retirement performs one host unmap");
    check(!mapper.get_buffer_mapping(a,found)&&mapper.get_pending_unmap_reqs().empty(),"pending retirement removes stale references");
    mapper.retire_buffer_mapping(a);check(unmaps.size()==1,"repeat retirement is no-op");
    mapper.retire_buffer_mapping(b);
    check(unmaps.size()==2&&unmaps[1]==12,"active retirement performs one host unmap");
    check(!mapper.get_buffer_mapping(b,found),"active retirement removes stale references");
    destination.buffer=cb;mapper.validate(&destination);check(maps==3,"LRU capacity/accounting valid after retirement");
    check(mapper.get_pending_unmap_reqs().empty(),"retired active capacity reclaimed");
    mapper.retire_buffer_mapping(c);check(unmaps.size()==3&&unmaps[2]==13,"new mapping can retire after reload");
    ggml_backend_buffer_free(ab);ggml_backend_buffer_free(bb);ggml_backend_buffer_free(cb);
    std::puts("PASS: production mapper active/pending retirement, idempotence, LRU accounting, reuse without stale refs");
}
