import AsideMenu from "./AsideMenu.jsx";
import AsideValidation from "./AsideValidation.jsx";


function AsideAdmin (props) {

    return <aside>
        <AsideMenu currentPage={props.currentPage} toFeedPage={props.toFeedPage} toFeedAdminPage={props.toFeedAdminPage} />
        <AsideValidation />
    </aside>;

}

export default AsideAdmin;