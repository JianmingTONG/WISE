#include "context.hpp"
#include "utils.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

inline constexpr auto SER_BIN = lbcrypto::SerType::BINARY;
namespace py = pybind11;
using namespace fheproj;

// numpy -> vector<double>
static std::vector<double> np_to_vec_double(py::handle obj) {
    py::array a = py::cast<py::array>(obj);
    py::array_t<double, py::array::c_style | py::array::forcecast> arr(a);
    auto info = arr.request();
    auto* p = static_cast<double*>(info.ptr);
    return std::vector<double>(p, p + info.size);
}

// numpy -> vector<vector<double>>
static std::vector<std::vector<double>> np_to_vec2d_double(py::handle obj) {
    py::array a = py::cast<py::array>(obj);
    py::array_t<double, py::array::c_style | py::array::forcecast> arr(a);
    auto info = arr.request();
    if (info.ndim != 2) throw py::value_error("bias must be a 2D float64 array");
    const size_t rows = static_cast<size_t>(info.shape[0]);
    const size_t cols = static_cast<size_t>(info.shape[1]);
    auto* p = static_cast<double*>(info.ptr);
    std::vector<std::vector<double>> out(rows);
    for (size_t r = 0; r < rows; ++r) { out[r].assign(p + r * cols, p + (r + 1) * cols); }
    return out;
}
// numpy -> vector<vector<T>>
template<class T> static std::vector<std::vector<T>> np_to_vec2d(py::handle obj) {
    py::array a = py::cast<py::array>(obj);
    py::array_t<T, py::array::c_style | py::array::forcecast> arr(a);

    auto info = arr.request();
    if (info.ndim != 2) throw py::value_error("expect a 2D array");

    const size_t rows = static_cast<size_t>(info.shape[0]);
    const size_t cols = static_cast<size_t>(info.shape[1]);

    auto* p = static_cast<T*>(info.ptr);
    std::vector<std::vector<T>> out(rows);
    for (size_t r = 0; r < rows; ++r) out[r].assign(p + r * cols, p + (r + 1) * cols);
    return out;
}

template<class T> static py::array to_ndarray_2d_generic(const std::vector<std::vector<T>>& M) {
    using ssize = py::ssize_t;

    const ssize rows = static_cast<ssize>(M.size());
    const ssize cols = rows ? static_cast<ssize>(M[0].size()) : 0;

    py::array_t<T> a({rows, cols});
    auto buf = a.template mutable_unchecked<2>();

    for (ssize r = 0; r < rows; ++r)
        for (ssize c = 0; c < cols; ++c) buf(r, c) = M[r][c];

    return a;
}

namespace pybind11::detail {
template<> struct type_caster<Diagonal> {
    PYBIND11_TYPE_CASTER(Diagonal, _("Diagonal"));
    bool load(handle src, bool) {
        if (!src) return false;
        py::object o = py::reinterpret_borrow<py::object>(src);
        if (!py::hasattr(o, "bs") || !py::hasattr(o, "gs") || !py::hasattr(o, "data")) return false;
        try {
            value.bs = py::cast<int>(o.attr("bs"));
            value.gs = py::cast<int>(o.attr("gs"));
            value.data = np_to_vec_double(o.attr("data"));
            return true;
        } catch (...) { return false; }
    }
    static handle cast(const Diagonal& d, return_value_policy, handle) {
        py::dict out;
        out["bs"] = d.bs;
        out["gs"] = d.gs;
        out["data"] = py::array_t<double>(d.data.size(), d.data.data());
        return out.release();
    }
};
}

namespace pybind11::detail {
template<> struct type_caster<Block> {
    PYBIND11_TYPE_CASTER(Block, _("Block"));
    bool load(handle src, bool) {
        if (!src) return false;
        py::object o = py::reinterpret_borrow<py::object>(src);
        if (!py::hasattr(o, "bx") || !py::hasattr(o, "by") || !py::hasattr(o, "diags")) return false;
        try {
            value.bx = py::cast<int>(o.attr("bx"));
            value.by = py::cast<int>(o.attr("by"));
            value.diags = py::cast<std::vector<Diagonal>>(o.attr("diags"));
            return true;
        } catch (...) { return false; }
    }
    static handle cast(const Block& b, return_value_policy, handle) {
        py::dict out;
        out["bx"] = b.bx;
        out["by"] = b.by;
        py::list L;
        for (const auto& d: b.diags) L.append(py::cast(d));
        out["diags"] = std::move(L);
        return out.release();
    }
};
}
namespace pybind11::detail {
template<> struct type_caster<HerPNReady> {
    PYBIND11_TYPE_CASTER(HerPNReady, _("HerPNReady"));
    bool load(handle src, bool) {
        if (!src) return false;
        py::object o = py::reinterpret_borrow<py::object>(src);
        if (!py::hasattr(o, "a0") || !py::hasattr(o, "a1")) return false;
        try {
            value.a0 = np_to_vec2d<double>(o.attr("a0"));
            value.a1 = np_to_vec2d<double>(o.attr("a1"));
            return true;
        } catch (...) { return false; }
    }
    static handle cast(const HerPNReady& v, return_value_policy, handle) {
        py::dict out;
        out["a0"] = to_ndarray_2d_generic<double>(v.a0);
        out["a1"] = to_ndarray_2d_generic<double>(v.a1);
        return out.release();
    }
};

template<> struct type_caster<NonlinearReady> {
    PYBIND11_TYPE_CASTER(NonlinearReady, _("NonlinearReady"));
    bool load(handle src, bool) {
        if (!src) return false;
        py::object o = py::reinterpret_borrow<py::object>(src);
        if (!py::hasattr(o, "coeffs") || !py::hasattr(o, "a") || !py::hasattr(o, "b") || !py::hasattr(o, "degree")) {
            return false;
        }

        try {
            py::array_t<double, py::array::c_style | py::array::forcecast> coeffs_arr(o.attr("coeffs"));
            auto req = coeffs_arr.request();
            const size_t n = static_cast<size_t>(req.size);
            value.coeffs.resize(n);
            if (n) std::memcpy(value.coeffs.data(), req.ptr, n * sizeof(double));

            value.a = py::cast<double>(o.attr("a"));
            value.b = py::cast<double>(o.attr("b"));
            value.degree = py::cast<int>(o.attr("degree"));

            return true;
        } catch (...) { return false; }
    }
    static handle cast(const NonlinearReady& v, return_value_policy, handle) {
        py::dict out;

        py::array_t<double> coeffs_arr(v.coeffs.size());
        if (!v.coeffs.empty()) { std::memcpy(coeffs_arr.mutable_data(), v.coeffs.data(), v.coeffs.size() * sizeof(double)); }
        out["coeffs"] = std::move(coeffs_arr);
        out["a"] = py::float_(v.a);
        out["b"] = py::float_(v.b);
        out["degree"] = py::int_(v.degree);
        return out.release();
    }
};

template<> struct type_caster<LinearTransformReady> {
    PYBIND11_TYPE_CASTER(LinearTransformReady, _("LinearTransformReady"));
    bool load(handle src, bool) {
        if (!src) return false;
        py::object o = py::reinterpret_borrow<py::object>(src);
        if (!py::hasattr(o, "T_rows") || !py::hasattr(o, "T_cols") || !py::hasattr(o, "output_rotations")
            || !py::hasattr(o, "mat") || !py::hasattr(o, "bias") || !py::hasattr(o, "v_bs") || !py::hasattr(o, "num_diags")
            || !py::hasattr(o, "global_rots"))
            return false;
        try {
            value.T_rows = py::cast<int>(o.attr("T_rows"));
            value.T_cols = py::cast<int>(o.attr("T_cols"));
            value.output_rotations = py::cast<int>(o.attr("output_rotations"));
            value.mat = py::cast<std::vector<std::vector<Block>>>(o.attr("mat"));
            value.bias = np_to_vec2d<double>(o.attr("bias"));
            value.v_bs = py::cast<std::vector<std::vector<int>>>(o.attr("v_bs"));
            value.num_diags = py::cast<int>(o.attr("num_diags"));
            value.global_rots.clear();
            for (auto item: py::cast<py::set>(o.attr("global_rots"))) value.global_rots.insert(py::cast<int>(item));
            return true;
        } catch (...) { return false; }
    }
    static handle cast(const LinearTransformReady& v, return_value_policy, handle) {
        py::dict out;
        out["T_rows"] = v.T_rows;
        out["T_cols"] = v.T_cols;
        out["output_rotations"] = v.output_rotations;

        py::list mat_outer;
        for (const auto& row: v.mat) {
            py::list row_list;
            for (const auto& b: row) row_list.append(py::cast(b));
            mat_outer.append(row_list);
        }
        out["mat"] = mat_outer;

        out["bias"] = to_ndarray_2d_generic<double>(v.bias);

        out["v_bs"] = v.v_bs;
        out["num_diags"] = v.num_diags;

        py::set gr;
        for (int x: v.global_rots) gr.add(py::int_(x));
        out["global_rots"] = gr;

        return out.release();
    }
};
template<> struct type_caster<fheproj::WinogradReady> {
    using T = fheproj::WinogradReady;
    PYBIND11_TYPE_CASTER(T, _("WinogradReady"));

    bool load(handle src, bool) {
        if (!src) return false;
        py::object o = py::reinterpret_borrow<py::object>(src);

        const char* req[] = {"T_rows",
                             "T_cols",
                             "output_rotations",
                             "R",
                             "S",
                             "Ht",
                             "Wt",
                             "i2c_target",
                             "i2c_offset",
                             "A_kron",
                             "B_kron",
                             "mats",
                             "bias",
                             "v_bs",
                             "num_diags",
                             "global_rots"};
        for (auto k: req)
            if (!py::hasattr(o, k)) return false;

        try {
            value.T_rows = py::cast<int>(o.attr("T_rows"));
            value.T_cols = py::cast<int>(o.attr("T_cols"));
            value.output_rotations = py::cast<int>(o.attr("output_rotations"));
            value.R = py::cast<int>(o.attr("R"));
            value.S = py::cast<int>(o.attr("S"));
            value.Ht = py::cast<int>(o.attr("Ht"));
            value.Wt = py::cast<int>(o.attr("Wt"));

            value.i2c_target = np_to_vec2d<int>(o.attr("i2c_target"));
            value.i2c_offset = np_to_vec2d<int>(o.attr("i2c_offset"));
            value.A_kron = np_to_vec2d<int>(o.attr("A_kron"));
            value.B_kron = np_to_vec2d<int>(o.attr("B_kron"));
            value.bias = np_to_vec2d<double>(o.attr("bias"));

            value.mats = py::cast<std::vector<std::vector<std::vector<fheproj::Block>>>>(o.attr("mats"));

            value.v_bs = py::cast<std::vector<std::vector<int>>>(o.attr("v_bs"));
            value.num_diags = py::cast<int>(o.attr("num_diags"));

            value.global_rots.clear();
            for (auto item: py::cast<py::set>(o.attr("global_rots"))) value.global_rots.insert(py::cast<int>(item));

            return true;
        } catch (...) { return false; }
    }

    static handle cast(const T& v, return_value_policy, handle) {
        py::dict out;
        out["T_rows"] = v.T_rows;
        out["T_cols"] = v.T_cols;
        out["output_rotations"] = v.output_rotations;
        out["R"] = v.R;
        out["S"] = v.S;
        out["Ht"] = v.Ht;
        out["Wt"] = v.Wt;

        out["i2c_target"] = to_ndarray_2d_generic(v.i2c_target);
        out["i2c_offset"] = to_ndarray_2d_generic(v.i2c_offset);
        out["A_kron"] = to_ndarray_2d_generic(v.A_kron);
        out["B_kron"] = to_ndarray_2d_generic(v.B_kron);

        // mats: list[list[list[Block]]]
        py::list mats_outer;
        for (const auto& row: v.mats) {
            py::list row_list;
            for (const auto& cell: row) {
                py::list cell_list;
                for (const auto& blk: cell) cell_list.append(py::cast(blk));
                row_list.append(std::move(cell_list));
            }
            mats_outer.append(std::move(row_list));
        }
        out["mats"] = std::move(mats_outer);

        out["bias"] = to_ndarray_2d_generic(v.bias);
        out["v_bs"] = v.v_bs;
        out["num_diags"] = v.num_diags;

        py::set gr;
        for (int x: v.global_rots) gr.add(py::int_(x));
        out["global_rots"] = std::move(gr);

        return out.release();
    }
};

}

PYBIND11_MODULE(fhecore, m) {
    m.doc() = "Map Python dataclasses to C++ structs; call real C++ in src/";

    py::class_<CiphertextHandle, std::shared_ptr<CiphertextHandle>>(m, "Ciphertext").def(py::init<>());

    py::class_<FHEContext>(m, "FHEContext")
        .def(py::init<const FHEContext::Params&>())
        .def("slots", &FHEContext::slots)
        .def("depth", &FHEContext::depth)
        .def(
            "encode_plaintext",
            [](const FHEContext& ctx, py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
                auto info = arr.request();
                auto* p = static_cast<double*>(info.ptr);
                std::vector<double> v(p, p + info.size);

                py::gil_scoped_release rel;
                std::string bytes = ctx.encode_plaintext_bytes(v);
                py::gil_scoped_acquire acq;

                return py::bytes(bytes);
            },
            py::arg("values"))

        // encrypt -> Ciphertext
        .def(
            "encrypt",
            [](const FHEContext& ctx, py::array_t<double, py::array::c_style | py::array::forcecast> arr, uint32_t level) {
                auto info = arr.request();
                if (static_cast<size_t>(info.size) != ctx.slots()) throw py::value_error("encrypt: len != slots");
                auto* p = static_cast<double*>(info.ptr);
                std::vector<double> v(p, p + info.size);
                py::gil_scoped_release rel;
                return ctx.encrypt(v, level);
            },
            py::arg("array"),
            py::arg("level"))
        // batch
        .def(
            "encrypt_batch",
            [](const FHEContext& ctx, py::array_t<double, py::array::c_style | py::array::forcecast> arr2d, uint32_t level) {
                auto info = arr2d.request();
                if (info.ndim != 2) throw py::value_error("encrypt_batch: expect 2D");
                size_t rows = info.shape[0], cols = info.shape[1];
                if (cols != ctx.slots()) throw py::value_error("encrypt_batch: cols != slots");
                auto* p = static_cast<double*>(info.ptr);
                std::vector<std::vector<double>> batch(rows);
                for (size_t r = 0; r < rows; ++r) batch[r] = {p + r * cols, p + (r + 1) * cols};
                py::gil_scoped_release rel;
                return ctx.encrypt_batch(batch, level);
            },
            py::arg("array2d"),
            py::arg("level"))
        // ctx.decrypt(handle, length) -> np.ndarray(shape=(length,), dtype=float64)
        .def(
            "decrypt",
            [](const FHEContext& ctx, const std::shared_ptr<CiphertextHandle>& h, size_t length) {
                py::gil_scoped_release rel;
                auto v = ctx.decrypt(h, length);
                py::gil_scoped_acquire acq;
                return py::array_t<double>(v.size(), v.data());
            },
            py::arg("ct"),
            py::arg("length") = 0)

        // ctx.decrypt_batch(list[handle], length) -> np.ndarray(shape=(B, length))
        .def(
            "decrypt_batch",
            [](const FHEContext& ctx, const std::vector<std::shared_ptr<CiphertextHandle>>& hs, size_t length) {
                py::gil_scoped_release rel;
                auto mats = ctx.decrypt_batch(hs, length);
                py::gil_scoped_acquire acq;
                size_t rows = mats.size();
                size_t cols = (length == 0) ? ctx.slots() : length;
                py::array_t<double> out({rows, cols});
                auto buf = out.mutable_unchecked<2>();
                for (size_t r = 0; r < rows; ++r) {
                    const auto& row = mats[r];
                    for (size_t c = 0; c < cols; ++c) buf(r, c) = (c < row.size() ? row[c] : 0.0);
                }
                return out;
            },
            py::arg("cts"),
            py::arg("length") = 0)
        .def(
            "deepcopy_ciphertexts",
            [](const FHEContext& ctx, const std::vector<std::shared_ptr<CiphertextHandle>>& xs) {
                py::gil_scoped_release rel;
                return ctx.deepcopy_ciphertexts(xs);
            },
            py::arg("cts"))
        .def(
            "stats",
            [](const FHEContext& ctx, const std::shared_ptr<CiphertextHandle>& h) {
                if (!h) throw py::value_error("stats: null handle");
                py::gil_scoped_release rel;
                auto [level, noise_deg, scale_bits, num_primes, rem] = ctx.stats(h);
                py::gil_scoped_acquire acq;
                return py::make_tuple(level, noise_deg, scale_bits, num_primes, rem);
            },
            py::arg("ct"))
        // c = a + b
        .def(
            "eval_add",
            [](const FHEContext& ctx, const std::shared_ptr<CiphertextHandle>& a, const std::shared_ptr<CiphertextHandle>& b) {
                if (!a || !b) throw py::value_error("eval_add: null handle");
                py::gil_scoped_release rel;
                return ctx.eval_add(a, b); // -> shared_ptr<CiphertextHandle>
            },
            py::arg("a"),
            py::arg("b"))

        // cs[i] = as[i] + bs[i]
        .def(
            "eval_add_batch",
            [](const FHEContext& ctx,
               const std::vector<std::shared_ptr<CiphertextHandle>>& as,
               const std::vector<std::shared_ptr<CiphertextHandle>>& bs) {
                if (as.size() != bs.size()) throw py::value_error("eval_add_batch: size mismatch");
                py::gil_scoped_release rel;
                return ctx.eval_add_batch(as, bs); // -> vector<shared_ptr<CiphertextHandle>>
            },
            py::arg("as"),
            py::arg("bs"))

        // a += b
        .def(
            "eval_add_in_place",
            [](const FHEContext& ctx, const std::shared_ptr<CiphertextHandle>& a, const std::shared_ptr<CiphertextHandle>& b) {
                if (!a || !b) throw py::value_error("eval_add_in_place: null handle");
                py::gil_scoped_release rel;
                ctx.eval_add_in_place(a, b);
            },
            py::arg("a"),
            py::arg("b"))

        // as[i] += bs[i]
        .def(
            "eval_add_in_place_batch",
            [](const FHEContext& ctx,
               const std::vector<std::shared_ptr<CiphertextHandle>>& as,
               const std::vector<std::shared_ptr<CiphertextHandle>>& bs) {
                if (as.size() != bs.size()) throw py::value_error("eval_add_in_place_batch: size mismatch");
                py::gil_scoped_release rel;
                ctx.eval_add_in_place_batch(as, bs);
            },
            py::arg("as"),
            py::arg("bs"))
        .def(
            "level_reduce",
            [](const FHEContext& ctx, const std::shared_ptr<CiphertextHandle>& a, size_t levels) {
                if (!a) throw py::value_error("level_reduce: null handle");
                py::gil_scoped_release rel;
                return ctx.level_reduce(a, levels);
            },
            py::arg("a"),
            py::arg("levels"))

        .def(
            "level_reduce_in_place",
            [](const FHEContext& ctx, const std::shared_ptr<CiphertextHandle>& a, size_t levels) {
                if (!a) throw py::value_error("level_reduce_in_place: null handle");
                py::gil_scoped_release rel;
                ctx.level_reduce_in_place(a, levels);
            },
            py::arg("a"),
            py::arg("levels"))

        .def(
            "level_reduce_batch",
            [](const FHEContext& ctx, const std::vector<std::shared_ptr<CiphertextHandle>>& as, size_t levels) {
                py::gil_scoped_release rel;
                return ctx.level_reduce_batch(as, levels);
            },
            py::arg("as"),
            py::arg("levels"))

        .def(
            "level_reduce_in_place_batch",
            [](const FHEContext& ctx, const std::vector<std::shared_ptr<CiphertextHandle>>& as, size_t levels) {
                py::gil_scoped_release rel;
                ctx.level_reduce_in_place_batch(as, levels);
            },
            py::arg("as"),
            py::arg("levels"))
        .def(
            "eval_linear_transform",
            [](const FHEContext& ctx,
               const std::vector<std::shared_ptr<CiphertextHandle>>& x,
               const fheproj::LinearTransformReady& lt,
               int target_level) {
                py::gil_scoped_release rel;
                return ctx.eval_linear_transform(x, lt, target_level);
            },
            py::arg("x"),
            py::arg("lt"),
            py::arg("target_levels_remaining"))
        .def(
            "eval_winograd",
            [](const FHEContext& ctx,
               const std::vector<std::shared_ptr<CiphertextHandle>>& x,
               const fheproj::WinogradReady& wino,
               int target_level) {
                py::gil_scoped_release rel;
                return ctx.eval_winograd(x, wino, target_level);
            },
            py::arg("x"),
            py::arg("wino"),
            py::arg("target_levels_remaining"))
        .def(
            "eval_herpn",
            [](const FHEContext& ctx,
               const std::vector<std::shared_ptr<CiphertextHandle>>& x,
               const fheproj::HerPNReady& hp,
               int target_level) {
                py::gil_scoped_release rel;
                return ctx.eval_herpn(x, hp, target_level);
            },
            py::arg("x"),
            py::arg("hp"),
            py::arg("target_levels_remaining"))
        .def(
            "eval_nonlinear",
            [](const FHEContext& ctx,
               const std::vector<std::shared_ptr<CiphertextHandle>>& x,
               const fheproj::NonlinearReady& nonlinear,
               int target_level) {
                py::gil_scoped_release rel;
                return ctx.eval_nonlinear(x, nonlinear, target_level);
            },
            py::arg("x"),
            py::arg("nonlinear"),
            py::arg("target_levels_remaining"))
        .def(
            "eval_winograd_advance",
            [](const FHEContext& ctx,
               const std::vector<std::shared_ptr<CiphertextHandle>>& x,
               const fheproj::WinogradReady& wino,
               int target_level) {
                py::gil_scoped_release rel;
                return ctx.eval_winograd(x, wino, target_level);
            },
            py::arg("x"),
            py::arg("wino"),
            py::arg("target_levels_remaining"))
        .def(
            "eval_bootstrap_batch",
            [](const FHEContext& ctx, const std::vector<std::shared_ptr<CiphertextHandle>>& x) {
                if (x.empty()) return std::vector<std::shared_ptr<CiphertextHandle>> {};
                for (size_t i = 0; i < x.size(); ++i) {
                    if (!x[i]) throw py::value_error("eval_bootstrap_batch: null handle at index " + std::to_string(i));
                }
                py::gil_scoped_release rel;
                return ctx.eval_bootstrap_batch(x);
            },
            py::arg("x"))
        .def("test",
             [](const FHEContext& ctx) {
                 py::gil_scoped_release rel;
                 ctx.test();
             })
        .def(
            "eval_silu_batch",
            [](const FHEContext& ctx,
               const std::vector<std::shared_ptr<CiphertextHandle>>& x,
               double a,
               double b,
               uint32_t degree) {
                if (x.empty()) return std::vector<std::shared_ptr<CiphertextHandle>> {};
                for (size_t i = 0; i < x.size(); ++i) {
                    if (!x[i]) throw py::value_error("eval_silu_batch: null handle at index " + std::to_string(i));
                }
                py::gil_scoped_release rel;
                return ctx.eval_silu_batch(x, a, b, degree);
            },
            py::arg("cts"),
            py::arg("a"),
            py::arg("b"),
            py::arg("degree"))

        ;

    py::class_<FHEContext::Params>(m, "FHEParams")
        .def(py::init<>())
        .def_readwrite("log_N", &FHEContext::Params::log_N)
        .def_readwrite("scale_mod_size", &FHEContext::Params::scale_mod_size)
        .def_readwrite("first_mod_size", &FHEContext::Params::first_mod_size)
        .def_readwrite("mult_depth", &FHEContext::Params::mult_depth)
        .def_readwrite("log_batch_size", &FHEContext::Params::log_batch_size)
        .def_readwrite("security_level", &FHEContext::Params::security_level)
        .def_readwrite("secret_key_dist", &FHEContext::Params::secret_key_dist)
        .def_readwrite("scaling_technique", &FHEContext::Params::scaling_technique)
        .def_readwrite("key_switch_technique", &FHEContext::Params::key_switch_technique)
        .def_readwrite("level_budget", &FHEContext::Params::level_budget) // std::vector<uint32_t>
        .def_readwrite("global_rots", &FHEContext::Params::global_rots) // std::vector<int32_t>
        .def_readwrite("composite_degree", &FHEContext::Params::composite_degree);
}
